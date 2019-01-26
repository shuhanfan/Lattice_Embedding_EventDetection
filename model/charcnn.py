# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2017-12-06 23:24:42
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class CNN(nn.Module):# N is batch_size, C is hidden_size, C_i means the channel number(initial is 1), W is the padding length, D means the dimensions
    def __init__(self,data, input_dim,hidden_dim,dropout):
        super(CNN, self).__init__()
        self.gloss_embeddings = nn.Embedding(data.gloss_alphabet.size(), data.gloss_emb_dim)
        self.gloss_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_gloss_embedding))
        D = input_dim
        C = hidden_dim
        Ci = 1
        Co = 100
        Ks = [2,3,4,5]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)
        
    def forward(self, x):
        x = self.dropout(self.gloss_embeddings(x))
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)# (N,len(Ks)*Co)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)return output    # return x for visualization
        return logit


class CharCNN(nn.Module):
    def __init__(self, data):
        super(CharCNN, self).__init__()
        print ("build batched gloss cnn...")
        self.HP_gpu = data.HP_gpu
        self.gloss_hidden_dim = data.gloss_hidden_dim
        self.gloss_drop = nn.Dropout(data.HP_dropout)
        self.gloss_embeddings = nn.Embedding(data.gloss_alphabet.size(), data.gloss_emb_dim)
        self.gloss_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_gloss_embedding))
        self.gloss_cnn = nn.Conv1d(data.gloss_emb_dim, data.gloss_hidden_dim, kernel_size=3, padding=1)
        if self.HP_gpu:
            self.gloss_drop = self.gloss_drop.cuda()
            self.gloss_embeddings = self.gloss_embeddings.cuda()
            self.gloss_cnn = self.gloss_cnn.cuda()



    def get_last_hiddens(self, input):
        """
            input:  
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output: 
                Variable(batch_size, self.gloss_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        gloss_embeds = self.gloss_drop(self.gloss_embeddings(input))
        gloss_embeds = gloss_embeds.transpose(2,1).contiguous()
        gloss_cnn_out = self.gloss_cnn(gloss_embeds)
        gloss_cnn_out = F.max_pool1d(gloss_cnn_out, gloss_cnn_out.size(2)).view(batch_size, -1)
        return gloss_cnn_out

    def get_all_hiddens(self, input, seq_lengths):
        """
            input:  
                input: Variable(batch_size,  word_length)
                seq_lengths: numpy array (batch_size,  1)
            output: 
                Variable(batch_size, word_length, gloss_self.gloss_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        gloss_embeds = self.gloss_drop(self.gloss_embeddings(input))
        gloss_embeds = gloss_embeds.transpose(2,1).contiguous()
        gloss_cnn_out = self.gloss_cnn(gloss_embeds).transpose(2,1).contiguous()
        return gloss_cnn_out



    def forward(self, input, seq_lengths):
        return self.get_all_hiddens(input, seq_lengths)
        