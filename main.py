from model.bilstmcrf import BiLSTM_CRF as SeqModel
import argparse
from data import Data
import time
import sys
import argparse
import random
import copy
import gc
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np



def batchify_with_label(input_batch_list, gpu, volatile_flag=False):
    """
        input: list of words, chars and labels, various length. [[words,entitys,gazs,glosses,labels],[words,entitys,gazs,glosses,labels],...]
            words: word ids for one sentence. (batch_size, sent_len) 
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order 
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len) 
    """
    #这里就是分别取出来
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    entitys = [sent[1] for sent in input_batch_list]
    gazs = [sent[2] for sent in input_batch_list]
    sentence_glosses = [sent[3] for sent in input_batch_list]
    labels = [sent[4] for sent in input_batch_list]
    #计算每个句子的长度
    word_seq_lengths = list(map(len, words))#这里会把每个训练句子的长度读出来
    max_seq_len = max(word_seq_lengths)
    #对word,entity,label,mask进行补零
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile =  volatile_flag).long()
    entity_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile =  volatile_flag).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).long()
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).byte()
    for idx, (seq, enseq, label, seqlen) in enumerate(zip(words, entitys, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        entity_seq_tensor[idx, :seqlen] = torch.LongTensor(enseq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
    #对sentence_gloss进行补零
    length_list = [list(map(len, glosses)) for glosses in sentence_glosses]#这是会把每个训练句子中每个字的解释长度读出来
    max_word_len = max(max(length_list))#把最长的解释读出来
    if max_word_len==0:
        max_word_len=10
    gloss_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len)), volatile =  volatile_flag).long()
    gloss_seq_lengths = torch.LongTensor(length_list)
    for idx, (glosses, glosseslen) in enumerate(zip(sentence_glosses, gloss_seq_lengths)):
        for idy, (gloss, glosslen) in enumerate(zip(glosses, glosseslen)):
            # print(gloss,glosslen)
            gloss_seq_tensor[idx, idy, :glosslen] = torch.LongTensor(gloss)
    
    #如果有GPU，转换一下形式
    gaz_list = [gazs,volatile_flag]
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        entity_seq_tensor = entity_seq_tensor.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        gloss_seq_tensor = gloss_seq_tensor.cuda()
        mask = mask.cuda()
    return gaz_list, word_seq_tensor, entity_seq_tensor, gloss_seq_tensor,label_seq_tensor, mask


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1-decay_rate)**epoch)
    print( " Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer



def evaluate_result(pred_variable, gold_variable, mask_variable, negid=0):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy().flatten()
    gold = gold_variable.cpu().data.numpy().flatten()
    mask = mask_variable.cpu().data.numpy().flatten()
    # print(predict.shape)
    gold_total, pred_total, pred_right = 0, 0, 0
    for index in range(pred.shape[0]):
        a=int(pred[index])
        b=int(gold[index])
        if a != negid: pred_total += 1
        if b != negid: gold_total += 1
        if a != negid and a == b: pred_right += 1
    # gold_total = 440
    ret = ('Pred_total: %d, Pred_right: %d, Gold_total: %d\n' % (pred_total, pred_right, gold_total))
    precision = 0.0 if pred_total == 0 else 1. * pred_right / pred_total
    recall = 0.0 if gold_total == 0 else 1. * pred_right / gold_total
    f = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    ret += ('P:%.2f, R:%.2f, F:%.2f' % (100 * precision, 100 * recall, 100 * f))
    print(ret)
    return precision

def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token




def train(data, save_model_dir, seg=True):
    model = SeqModel(data)
    # print "finished built model."
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.SGD(parameters, lr=data.HP_lr, momentum=data.HP_momentum)
    ## start training
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" %(idx,data.HP_iteration))
        # random.shuffle(data.index_data)
        ## set model in train model
        model.train()
        model.zero_grad()
        batch_size = 1 ## current only support batch size = 1 to compulate and accumulate to data.HP_batch_size update weights
        train_num = len(data.index_data)
        total_batch = train_num//batch_size+1
        for batch_id in range(total_batch):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size 
            if end >train_num:
                end = train_num
            instance = data.index_data[start:end]
            if not instance:
                continue
            gaz_list, batch_word, batch_entity, batch_gloss, batch_label, mask = batchify_with_label(instance, data.HP_gpu)
            loss,predict = model.neg_log_likelihood_loss(gaz_list,batch_word,batch_entity,batch_gloss,batch_label, mask)
            # print(predict)
            print("Batch_Id:",batch_id," Loss:",loss)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            predict_check(predict, batch_label, mask)
            evaluate_result(predict, batch_label, mask)
        print("\n")





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with bi-directional LSTM-CRF')
    parser.add_argument('--embedding',  help='Embedding for words', default='None')
    parser.add_argument('--status', choices=['train', 'test', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--savemodel', default="data/model/saved_model.lstmcrf.")
    parser.add_argument('--savedset', help='Dir of saved data setting', default="data/save.dset")
    parser.add_argument('--train', default="dataset/data_middle_smaller.json") 
    parser.add_argument('--dev', default="data/conll03/dev.bmes" )  
    parser.add_argument('--test', default="data/conll03/test.bmes") 
    parser.add_argument('--seg', default="True") 
    parser.add_argument('--extendalphabet', default="True") 
    parser.add_argument('--raw') 
    parser.add_argument('--loadmodel')
    parser.add_argument('--output') 
    args = parser.parse_args()
   
    train_file = args.train
    dev_file = args.dev
    test_file = args.test
    raw_file = args.raw
    model_dir = args.loadmodel
    dset_dir = args.savedset
    output_file = args.output
    if args.seg.lower() == "true":
        seg = True 
    else:
        seg = False
    status = args.status.lower()

    save_model_dir = args.savemodel
    gpu = torch.cuda.is_available()

    # char_emb = "data/gigaword_chn.all.a2b.uni.ite50.vec"
    # bichar_emb = None
    # gaz_file = "data/ctb.50d.vec"
    # gaz_file = None
    # char_emb = None
    #bichar_emb = None

    # print "CuDNN:", torch.backends.cudnn.enabled
    # # gpu = False
    # print "GPU available:", gpu
    # print "Status:", status
    # print "Seg: ", seg
    # print "Train file:", train_file
    # print "Dev file:", dev_file
    # print "Test file:", test_file
    # print "Raw file:", raw_file
    # print "Char emb:", char_emb
    # print "Bichar emb:", bichar_emb
    # print "Gaz file:",gaz_file
    # if status == 'train':
    #     print "Model saved to:", save_model_dir
    # sys.stdout.flush()
    
    if status == 'train':
        data = Data(train_file)
        data.build_alphabet()
        data.generate_instance_Ids()
        data.generate_embedding()
        train(data, save_model_dir)
    elif status == 'test':      
        data = load_data_setting(dset_dir)
        data.generate_instance_with_gaz(dev_file,'dev')
        load_model_decode(model_dir, data , 'dev', gpu, seg)
        data.generate_instance_with_gaz(test_file,'test')
        load_model_decode(model_dir, data, 'test', gpu, seg)
    elif status == 'decode':       
        data = load_data_setting(dset_dir)
        data.generate_instance_with_gaz(raw_file,'raw')
        decode_results = load_model_decode(model_dir, data, 'raw', gpu, seg)
        data.write_decoded_results(output_file, decode_results, 'raw')
    else:
        print ("Invalid argument! Please use valid arguments! (train/test/decode)")



