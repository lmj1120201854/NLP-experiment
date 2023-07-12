import torch
from torch.utils.data import Dataset, DataLoader
import json
import Exp3_Config


def get_pos_index(x):  # 判断相对距离是否有效，并且将距离规范化
    if x < -50:
        return 0
    if -50 <= x <= 50:
        return x + 50 + 1
    if x > 50:
        return 102


def get_relative_pos(x, ep):  # 计算相对距离
    return get_pos_index(x - ep)


def trans_json():  # 导入字符和标签的映射字典
    with open('data/rel2id.json', 'r', encoding='utf8') as fp:
        tem = json.load(fp)
        rel2id = tem[1]
        id2rel = tem[0]
    with open('data/vocab.json', 'r', encoding='utf8') as fp:
        tem = json.load(fp)
        word2token = tem[1]
        token2word = tem[0]
    return rel2id, id2rel, word2token, token2word


# 训练集和验证集
class TextDataSet(Dataset):
    def __init__(self, filepath, config, rel2id=None, id2rel=None, word2token=None, token2word=None):
        self.config = config
        self.rel2id = rel2id
        self.id2rel = id2rel
        self.word2token = word2token
        self.token2word = token2word

        def pad(origin_sent):  # 补齐函数，将token补齐或剪裁到规定长度，使得能够转化成tensor的形式打包成数据集
            if len(origin_sent) > self.config.max_sentence_length:
                return origin_sent[:self.config.max_sentence_length]
            else:
                return origin_sent + [0 for _ in range(self.config.max_sentence_length - len(origin_sent))]

        lines = open(filepath, 'r', encoding='utf-8').readlines()
        self.original_data = []
        for line in lines:
            tmp = {}
            line = line.split('\t')
            tmp['head'] = line[3].find(line[0])
            tmp['tail'] = line[3].find(line[1])
            '''tmp['relation'] = line[2]
            tmp['text'] = line[3]'''
            tmp['relation'] = self.rel2id[line[2]]
            tok = []
            for word in line[3][:-1]:
                tok.append(self.word2token[word])  # 将字符转化为序号序列
            tmp['text'] = torch.tensor(pad(tok))
            pos1 = []
            pos2 = []
            for pos in range(len(tmp['text'])):
                pos1.append(get_relative_pos(pos, int(tmp['head'])))  # 计算每个词到首实体的相对距离
                pos2.append(get_relative_pos(pos, int(tmp['tail'])))  # 计算每个词到尾实体的相对距离
            tmp['pos1'] = torch.tensor(pos1)
            tmp['pos2'] = torch.tensor(pos2)
            self.original_data.append(tmp)

    def __getitem__(self, index):
        return self.original_data[index]

    def __len__(self):
        return len(self.original_data)


# 测试集是没有标签的，因此函数会略有不同
class TestDataSet(Dataset):
    def __init__(self, filepath, config, word2token=None, token2word=None):
        self.config = config
        self.word2token = word2token
        self.token2word = token2word

        def pad(origin_sent):
            if len(origin_sent) > self.config.max_sentence_length:
                return origin_sent[:self.config.max_sentence_length]
            else:
                return origin_sent + [0 for _ in range(self.config.max_sentence_length - len(origin_sent))]

        lines = open(filepath, 'r', encoding='utf-8').readlines()
        self.original_data = []
        for line in lines:
            tmp = {}
            line = line.split('\t')
            tmp['head'] = line[2].find(line[0])
            tmp['tail'] = line[2].find(line[1])
            '''tmp['text'] = line[2]'''
            tok = []
            for word in line[2][:-1]:
                tok.append(self.word2token[word])
            tmp['text'] = torch.tensor(pad(tok))
            pos1 = []
            pos2 = []
            for pos in range(len(tmp['text'])):
                pos1.append(get_relative_pos(pos, int(tmp['head'])))
                pos2.append(get_relative_pos(pos, int(tmp['tail'])))
            tmp['pos1'] = torch.tensor(pos1)
            tmp['pos2'] = torch.tensor(pos2)
            self.original_data.append(tmp)

    def __getitem__(self, index):
        return self.original_data[index]

    def __len__(self):
        return len(self.original_data)


if __name__ == "__main__":
    config = Exp3_Config.Training_Config()
    rel2id, id2rel, word2token, token2word = trans_json()
    '''print(rel2id, id2rel)
    print(word2token, token2word)'''
    trainset = TextDataSet(filepath="./data/data_train.txt", config=config,
                           rel2id=rel2id, id2rel=id2rel, word2token=word2token, token2word=token2word)
    testset = TestDataSet(filepath="./data/test_exp3.txt", config=config, word2token=word2token, token2word=token2word)
    print("训练集长度为：", len(trainset))
    print("测试集长度为：", len(testset))
    a = DataLoader(dataset=trainset, batch_size=1)
    for i in a:
        # print("测试集长度为：", i['text'])
        print(i['head'], i['tail'])
        '''if len(i['text'][0]) > max:
            max = len(i['text'][0])
    print(max)'''

    # trans_json()

