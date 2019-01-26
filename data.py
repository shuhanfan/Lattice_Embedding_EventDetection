import json
from alphabet import Alphabet
import numpy as np
class Data:
    def __init__(self,input_file): 
        self.original_data = open(input_file,'r').readlines()
        self.index_data = []
        self.word_alphabet = Alphabet('word')
        self.gloss_alphabet = Alphabet('gloss')
        self.entity_alphabet = Alphabet('entity')
        self.gaz_alphabet = Alphabet('gaz')
        self.label_alphabet = Alphabet('label')
        self.word_alphabet_size = 0
        self.gloss_alphabet_size = 0
        self.entity_alphabet_size = 0
        self.gaz_alphabet_size = 0
        self.label_alphabet_size = 0
        ### hyperparameters
        self.HP_iteration = 100
        self.HP_batch_size = 1
        self.HP_gaz_hidden_dim = 50
        self.HP_lstm_hidden_dim = 200
        self.HP_dropout = 0.5
        self.gaz_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = False
        self.HP_use_entity = False
        self.HP_use_gloss = True
        self.HP_use_gaz = False
        self.HP_gpu = True
        self.HP_lr = 0.015
        self.HP_lr_decay = 0.05
        self.HP_clip = 5.0
        self.HP_momentum = 0
        self.HP_iteration = 100
        # embedding hyperparameter
        self.word_emb_dim = 200
        self.entity_emb_dim = 50
        self.gloss_features = "CNN"#["CNN","LSTM"]
        self.gloss_emb_dim = 200
        self.gloss_hidden_dim = 300
        self.pretrain_word_embedding = np.array([])
        self.pretrain_gaz_embedding = None
        self.word_embed_path = "../LOVECC/NYM.6B.200d.txt"#"NYM_200.txt"
        self.gaz_embed_path = None
        self.gaz_emb_dim = 200
        self.HP_fix_gaz_emb = True


    def build_alphabet(self):
        in_lines = self.original_data
        for idx in range(len(in_lines)):
            line = json.loads(in_lines[idx])
            words = line["word_context"]
            for word in words:
                self.word_alphabet.add(word)

            sentence_gloss = line["babel_gloss"]
            for word_gloss in sentence_gloss:
                for phrase_gloss in word_gloss:#一个词可以匹配多个词组
                    if "EN" in phrase_gloss:
                        phrase_gloss_EN = phrase_gloss["EN"]
                        final_gloss=" . ".join(phrase_gloss_EN)
                        for de_word in final_gloss:
                        # for definates in phrase_gloss_EN:
                            # for de_word in definates.split():
                            self.gloss_alphabet.add(de_word)

            entitys = line["entity_context"]
            for entity in entitys:
                self.entity_alphabet.add(entity)

            gazs= line["babel_phase"]
            for gaz in gazs:
                for item in gaz:
                    self.gaz_alphabet.add(item)

            labels = line["detection_label"]
            for label in labels:
                self.label_alphabet.add(label)
        print(self.label_alphabet.get_content())
        self.word_alphabet_size = self.word_alphabet.size()
        self.gloss_alphabet_size = self.gloss_alphabet.size()
        self.entity_alphabet_size = self.entity_alphabet.size()
        self.gaz_alphabet_size = self.gaz_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()
        self.word_alphabet.close()
        self.gloss_alphabet.close()
        self.entity_alphabet.close()
        self.gaz_alphabet.close()
        self.label_alphabet.close()


    def generate_instance_Ids(self):#把输入句子变成对应的标号（Id）
        in_lines = self.original_data
        for idx in range(len(in_lines)):
            line = json.loads(in_lines[idx])
            words = line["word_context"]
            words_Id = []
            for word in words:
                words_Id.append(self.word_alphabet.get_index(word))

            sentence_gloss = line["babel_gloss"]
            sentence_glosses_Id = []
            for word_gloss in sentence_gloss:
                word_glosses_Id = []
                for phrase_gloss in word_gloss:#一个词可以匹配多个词组
                    if "EN" in phrase_gloss:
                        phrase_gloss_EN = phrase_gloss["EN"]#这是个list
                        final_gloss=" . ".join(phrase_gloss_EN)
                        for de_word in final_gloss:
                            word_glosses_Id.append(self.gloss_alphabet.get_index(de_word))
                sentence_glosses_Id.append(word_glosses_Id)
                            
            
            entitys = line["entity_context"]
            entitys_Id = []
            for entity in entitys:
                entitys_Id.append(self.entity_alphabet.get_index(entity))

            gazs = line["babel_phase"]
            sentence_gazs_Id= []#gazs_Id=[[[take over,take over of,...],[2,3,...]],[[legal,legal procedures,...],[1,2,...]],...,[[open the window,open the window please,...],[3,4,...]]]
            for gaz in gazs:
                word_gazs_Id=[]
                Ids=[]
                Lens=[]
                for item in gaz:
                    Ids.append(self.gaz_alphabet.get_index(item))
                    Lens.append(len(item.split()))
                word_gazs_Id=[Ids,Lens]
                sentence_gazs_Id.append(word_gazs_Id)

            labels = line["detection_label"]
            labels_Id = []
            for label in labels:
                labels_Id.append(self.label_alphabet.get_index(label))
            self.index_data.append([words_Id,entitys_Id,sentence_gazs_Id,sentence_glosses_Id,labels_Id])



    def load_pretrain_emb(self,embedding_path):
        lines=open(embedding_path, 'r',encoding="utf-8").readlines()
        statistic=lines[0].strip()#开头的两个统计数据：单词数，向量长度
        # print(statistic)
        embedd_dim = int(statistic.split()[1])
        embedd_dict = dict()
        embedd_dict["<pad>"]=[0.0 for i in range(embedd_dim)]#填充词对应的向量置为全零
        # print(len(embedd_dict["<pad>"]))
        for line in lines[1:]:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd_dict[tokens[0]] = [float(i) for i in tokens[1:]]
        return embedd_dict, embedd_dim


    def norm2one(self,vec):
        if np.sum(vec)==0:
            return vec
        root_sum_square = np.sqrt(np.sum(np.square(vec)))
        return vec/root_sum_square

    def build_pretrain_embedding(self,embedding_path, word_alphabet, embedd_dim=200, norm=True):
        embedd_dict = dict()
        if embedding_path != None:
            # 读取embedding字典
            embedd_dict, embedd_dim = self.load_pretrain_emb(embedding_path)
        scale = np.sqrt(3.0 / embedd_dim)
        pretrain_emb = np.zeros([word_alphabet.size(), embedd_dim])#pretrain_emb就是重排之后的embedding矩阵
        perfect_match = 0
        case_match = 0
        not_match = 0
        for word,index in word_alphabet.get_alphabet().items():
            if word in embedd_dict:
                # print(word,index)
                # print(len(embedd_dict[word]))
                if norm:
                    pretrain_emb[index] = self.norm2one(embedd_dict[word])
                else:
                    pretrain_emb[index] = embedd_dict[word]
                perfect_match += 1
            elif word.lower() in embedd_dict:
                if norm:
                    pretrain_emb[index] = self.norm2one(embedd_dict[word.lower()])
                else:
                    pretrain_emb[index] = embedd_dict[word.lower()]
                case_match += 1
            else:
                pretrain_emb[index] = np.random.uniform(-scale, scale, [1, embedd_dim])
                not_match += 1
        pretrained_size = len(embedd_dict)
        # print("pad's embedding:",pretrain_emb[word_alphabet.get_index(",")])
        print("Embedding:\n  pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/word_alphabet.size()))
        return pretrain_emb, embedd_dim #pretrain_emb就是根据alphabet的顺序重排embedding矩阵，embedd_dim是向量的纬度


    def generate_embedding(self):
        self.pretrain_word_embedding,self.word_pretrain_dim = self.build_pretrain_embedding(self.word_embed_path,self.word_alphabet)
        self.pretrain_gloss_embedding,self.gloss_pretrain_dim = self.build_pretrain_embedding(self.word_embed_path,self.gloss_alphabet)
        self.pretrain_gaz_embedding,self.gaz_pretrain_dim = self.build_pretrain_embedding(self.word_embed_path,self.gaz_alphabet)


# build_alphabet(data, data_file)
# generate_instance_Ids(data_file)
# build_word_pretrain_emb(emb_file)
