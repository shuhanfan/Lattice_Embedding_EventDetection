

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.bilstm import BiLSTM
# from model.crf import CRF
import torch.nn.functional as Fun

class BiLSTM_CRF(nn.Module):
    def __init__(self, data):
        super(BiLSTM_CRF, self).__init__()
        print ("build batched lstmcrf...")
        self.gpu = data.HP_gpu
        ## add two more label for downlayer lstm, use original label size for CRF
        self.lstm = BiLSTM(data)
        self.softmax = nn.LogSoftmax()
        self.loss_op = nn.CrossEntropyLoss()
        self.negid = data.label_alphabet.get_index("NEGATIVE")
        # self.crf = CRF(label_size, self.gpu)

    def count_weight_loss(self,output,targets):
        #计算loss,排除negtive这一类
        ONLY_POSITIVE=False
        targets_tensor = targets#torch.from_numpy(targets).type(torch.LongTensor)
        softmax_result = Fun.log_softmax(output)
        log_loss = softmax_result[torch.arange(targets_tensor.shape[0]).type(torch.LongTensor),targets_tensor]
        if ONLY_POSITIVE:
            classify_loss = -torch.mean(log_loss)
        else:
            if (targets_tensor!=self.negid).nonzero().shape[0]==0:#如果只有negid
                classify_loss = -torch.mean(1*log_loss[(targets_tensor==self.negid).nonzero()])
            else:
                classify_loss = -torch.mean(log_loss[(targets_tensor!=self.negid).nonzero()])-torch.mean(1*log_loss[(targets_tensor==self.negid).nonzero()])#(targets_tensor != label_dict["NEGATIVE"])
        l2_reg = Variable(torch.FloatTensor([0]), requires_grad=True)
        loss = Variable(torch.FloatTensor([0]), requires_grad=True)
        if args.cuda:
            l2_reg= l2_reg.cuda()
            loss= loss.cuda()
        for W in filter(lambda p: p.requires_grad,model.parameters()):
            l2_reg += W.norm(2)
        loss = classify_loss + 1e-5 * l2_reg#args.l2_weight
        loss = loss.squeeze()
        return loss

    def neg_log_likelihood_loss(self,gaz_list,batch_word,batch_entity,batch_gloss,batch_label, mask):
        outs = self.lstm.get_output_score(gaz_list,batch_word,batch_entity,batch_gloss,batch_label, mask)
        outs = outs.view([-1,outs.shape[2]])
        batch_label = batch_label.view([-1])
        # batch_label = torch.zeros(batch_label.shape[0], batch_label.shape[1], outs.shape[2]).scatter_(2, batch_label.unsqueeze(-1).type(torch.LongTensor), 1)
        loss = self.loss_op(outs,batch_label).sum()
        # loss = self.count_weight_loss(outs,batch_label).sum()
        return loss,torch.max(outs, 1)[1]

