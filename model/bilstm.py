# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-05-03 21:58:36
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from kblayer import GazLayer
from model.charbilstm import CharBiLSTM
from model.charcnn import CharCNN,CNN
from model.latticelstm import LatticeLSTM

class BiLSTM(nn.Module):
    def __init__(self, data):
        super(BiLSTM, self).__init__()
        print ("build batched bilstm...")
        self.gpu = data.HP_gpu
        self.use_gloss = data.HP_use_gloss
        self.use_entity = data.HP_use_entity
        self.use_gaz = data.HP_use_gaz
        self.batch_size = data.HP_batch_size
        self.gloss_hidden_dim = 0
        self.embedding_dim = data.word_emb_dim
        self.gloss_hidden_dim = data.gloss_hidden_dim
        self.gloss_drop = data.HP_dropout
        self.drop = nn.Dropout(data.HP_dropout)
        self.word_embeddings = nn.Embedding(data.word_alphabet.size(), self.embedding_dim)
        if data.pretrain_word_embedding is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.embedding_dim)))
        if self.use_entity:
            self.entity_embeddings = nn.Embedding(data.entity_alphabet.size(), data.entity_emb_dim)
            self.entity_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(data.entity_alphabet.size(), data.entity_emb_dim)))
        if self.use_gloss:
            self.gloss_hidden_dim = data.gloss_hidden_dim
            self.gloss_embedding_dim = data.gloss_emb_dim
            if data.gloss_features == "CNN":
                self.gloss_feature = CNN(data,input_dim=data.gloss_emb_dim,hidden_dim=self.gloss_hidden_dim,dropout=self.gloss_drop)
                # self.gloss_feature = CharCNN(data)#data.gloss_alphabet.size(), self.gloss_embedding_dim, self.gloss_hidden_dim, data.HP_dropout, self.gpu)
            elif data.gloss_features == "LSTM":
                self.gloss_feature = CharBiLSTM(data.gloss_alphabet.size(), self.gloss_embedding_dim, self.gloss_hidden_dim, data.HP_dropout, self.gpu)
            else:
                print ("Error gloss feature selection, please check parameter data.gloss_features (either CNN or LSTM).")
                exit(0)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        self.droplstm = nn.Dropout(data.HP_dropout)
        if self.bilstm_flag:
            lstm_hidden_dim = data.HP_lstm_hidden_dim // 2
        else:
            lstm_hidden_dim = data.HP_lstm_hidden_dim
        lstm_input_dim = self.embedding_dim + self.gloss_hidden_dim
        self.forward_lstm = LatticeLSTM(lstm_input_dim, lstm_hidden_dim, data.gaz_dropout, data.gaz_alphabet.size(), data.gaz_emb_dim, data.pretrain_gaz_embedding, left2right=True, fix_word_emb=data.HP_fix_gaz_emb, gpu=self.gpu)
        if self.bilstm_flag:
            self.backward_lstm = LatticeLSTM(lstm_input_dim, lstm_hidden_dim, data.gaz_dropout, data.gaz_alphabet.size(), data.gaz_emb_dim, data.pretrain_gaz_embedding, left2right=False, fix_word_emb=data.HP_fix_gaz_emb, gpu=self.gpu)
        # self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden_dim, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(data.HP_lstm_hidden_dim, data.label_alphabet_size)

        if self.gpu:
            self.drop = self.drop.cuda()
            self.droplstm = self.droplstm.cuda()
            self.word_embeddings = self.word_embeddings.cuda()
            if self.use_entity:
                self.entity_embeddings = self.entity_embeddings.cuda()
            self.forward_lstm = self.forward_lstm.cuda()
            if self.bilstm_flag:
                self.backward_lstm = self.backward_lstm.cuda()
            self.hidden2tag = self.hidden2tag.cuda()


    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb



    def get_lstm_features(self,gaz_list,batch_word,batch_entity,batch_gloss,batch_label, mask):
        """
            input:
                batch_word: (batch_size, sent_len)
                gaz_list:
                word_seq_lengths: list of batch_size, (batch_size,1)
                gloss_inputs: (batch_size*sent_len, word_length)
                gloss_seq_lengths: list of whole batch_size for gloss, (batch_size*sent_len, 1)
                gloss_seq_recover: variable which records the gloss order information, used to recover gloss order
            output: 
                Variable(sent_len, batch_size, hidden_dim)
        """
        batch_size = batch_word.size(0)
        sent_len =batch_word.size(1)
        word_embs =  self.word_embeddings(batch_word)
        if self.gpu:
            word_embs = word_embs.cuda()
        if self.use_entity:
            entity_embs = self.entity_embeddings(batch_entity)
            if self.gpu:
                entity_embs = entity_embs.cuda()
            word_embs = torch.cat([word_embs, entity_embs],2)
        if self.use_gloss:
            ## calculate gloss lstm last hidden
            gloss_features = self.gloss_feature(np.reshape(batch_gloss,[-1,batch_gloss.shape[2]]))#################
            gloss_features = gloss_features.view(batch_size,sent_len,-1)
            ## concat word and gloss together
            if self.gpu:
                gloss_features = gloss_features.cuda()
            word_embs = torch.cat([word_embs, gloss_features], 2)
        word_embs = self.drop(word_embs)
        # lstm_out=[]
        # for bi in range(batch_size):
        forward_hidden = None
        lstm_out, hidden = self.forward_lstm(word_embs, gaz_list, forward_hidden)
        # lstm_out.append(lstm_out_bi)
        if self.bilstm_flag:
            backward_hidden = None 
            backward_lstm_out, backward_hidden = self.backward_lstm(word_embs, gaz_list, backward_hidden)
            lstm_out=torch.cat([lstm_out, backward_lstm_out],2)
        lstm_out = self.droplstm(lstm_out)
        return lstm_out



    def get_output_score(self,gaz_list,batch_word,batch_entity,batch_gloss,batch_label, mask):
        lstm_out = self.get_lstm_features(gaz_list,batch_word,batch_entity,batch_gloss,batch_label,mask)
        ## lstm_out (batch_size, sent_len, hidden_dim)
        outputs = self.hidden2tag(lstm_out)
        return outputs
    

    def neg_log_likelihood_loss(self, gaz_list, word_inputs, entity_inputs, word_seq_lengths, gloss_inputs, gloss_seq_lengths, gloss_seq_recover, batch_label, mask):
        ## mask is not used
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        total_word = batch_size * seq_len
        loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
        outs = self.get_output_score(gaz_list, word_inputs, entity_inputs, word_seq_lengths, gloss_inputs, gloss_seq_lengths, gloss_seq_recover)
        # outs (batch_size, seq_len, label_vocab)
        outs = outs.view(total_word, -1)
        score = F.log_softmax(outs, 1)
        loss = loss_function(score, batch_label.view(total_word))
        _, tag_seq  = torch.max(score, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        return loss, tag_seq


    def forward(self, gaz_list, word_inputs, entity_inputs, word_seq_lengths,  gloss_inputs, gloss_seq_lengths, gloss_seq_recover, mask):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        total_word = batch_size * seq_len
        outs = self.get_output_score(gaz_list,  word_inputs, entity_inputs, word_seq_lengths, gloss_inputs, gloss_seq_lengths, gloss_seq_recover)
        outs = outs.view(total_word, -1)
        _, tag_seq  = torch.max(outs, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        ## filter padded position with zero
        decode_seq = mask.long() * tag_seq
        return decode_seq




