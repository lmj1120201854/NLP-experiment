import re

import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import Counter
from config import Config
import tqdm
import time

config = Config()
PAD_TOKEN = config.PAD_TOKEN
NAME_TOKEN = config.NAME_TOKEN
NEAR_TOKEN = config.NEAR_TOKEN


def str2dict(str_list):
    dict_list = []
    map_os = list(map(lambda x: x.split(', '), str_list))
    # map_os -> [['A[a]', 'B[b]', ...], ['A[a]', 'B[b]', ...], ...]
    for map_o in map_os:
        # map_o -> ['A[a]', 'B[b]', ...]
        _dict = {}
        for item in map_o:
            key = item.split('[')[0]
            value = item.split('[')[1].replace(']', '')
            _dict[key] = value
        dict_list.append(_dict)
    return dict_list


class Tokenizer:
    def __init__(self, token2id):
        self.token2id = token2id
        self.id2token = {value: key for key, value in self.token2id.items()}
        self.vocab_size = len(self.token2id)

    def encode(self, words):
        token_ids = [self.token2id['[BOS]']]
        for word in words:
            token_ids.append(self.token2id.get(word, self.token2id['[UNK]']))
        token_ids.append(self.token2id['[EOS]'])
        return token_ids

    def decode(self, token_ids):
        words = []
        for token in token_ids:
            word = self.id2token[token]
            if word in ['[BOS]', '[EOS]', '[PAD]']:
                continue
            words.append(word)
        return ' '.join(words)


class E2EDataset(Dataset):
    def __init__(self, path, mode='train', field_tokenizer=None, tokenizer=None, max_src_len=80, max_target_len=80):
        super(E2EDataset, self).__init__()
        self.mode = mode
        self.max_src_len = max_src_len
        self.max_target_len = max_target_len
        # field_token
        self.field_tokenizer = field_tokenizer
        self.key_num = 0
        self.tokenizer = tokenizer
        # preprocess
        self.raw_data_x = []
        self.raw_data_y = []
        self.lexicalizations = []
        self.muti_data_y = {}
        self.padding = 0

        df = pd.read_csv(path)

        if mode == 'train' or mode == 'dev':
            self.mr = str2dict(df['mr'].values.tolist())
            self.ref = df['ref'].values.tolist()

        elif mode == 'test':
            self.mr = str2dict(df['MR'].values.tolist())
            self.ref = ['' for _ in range(len(self.mr))]

        if mode == 'train':
            self.create_field()
            self.preprocess()
            self.create_voc()
        else:
            assert field_tokenizer is not None and tokenizer is not None
            self.field_tokenizer = field_tokenizer
            self.key_num = len(self.field_tokenizer)
            self.tokenizer = tokenizer
            self.preprocess()

    def create_field(self):
        mr_key = list(map(lambda x: list(x.keys()), self.mr))  # 获取所有的属性
        counter = Counter()
        for line in mr_key:
            counter.update(line)
        # 按词频对属性进行排序，相当于位置编码
        _tokens = [(token, count) for token, count in counter.items()]
        _tokens = sorted(_tokens, key=lambda x: -x[1])
        # 获取词列表
        _tokens = [token for token, count in _tokens]
        # 创建word2token
        self.field_tokenizer = dict(zip(_tokens, range(len(_tokens))))
        self.key_num = len(self.field_tokenizer)

    def preprocess(self):
        for index in range(len(self.ref)):
            # 对mr数据进行预处理，补全属性、进行序列化
            mr_data = [PAD_TOKEN] * self.key_num
            lex = ['', '']
            for item in self.mr[index].items():
                key = item[0]
                value = item[1]
                key_idx = self.field_tokenizer[key]
                # 单词去重
                if key == 'name':
                    mr_data[key_idx] = NAME_TOKEN
                    lex[0] = value
                elif key == 'near':
                    mr_data[key_idx] = NEAR_TOKEN
                    lex[1] = value
                else:
                    mr_data[key_idx] = value
            # 对目标序列进行序列化
            ref_data = self.ref[index]
            if ref_data == '':
                ref_data = ['']
            else:
                if lex[0]:
                    ref_data = ref_data.replace(lex[0], NAME_TOKEN)
                if lex[1]:
                    ref_data = ref_data.replace(lex[1], NEAR_TOKEN)
                ref_data = list(map(lambda x: re.split(r"([.,!?\"':;)(])", x)[0],
                                    ref_data.split()))
            # 将处理好的每行mr和ref收录归纳
            self.raw_data_x.append(mr_data)
            self.raw_data_y.append(ref_data)
            self.lexicalizations.append(lex)
            # 多参考文本
            mr_data_str = ''.join(mr_data)
            if mr_data_str in self.muti_data_y.keys():
                self.muti_data_y[mr_data_str].append(self.ref[index])
            else:
                self.muti_data_y[mr_data_str] = [self.ref[index]]

    def create_voc(self):
        # 统计词频
        counter = Counter()
        for line in self.raw_data_x:
            counter.update(line)
        for line in self.raw_data_y:
            counter.update(line)
        # 排序
        _tokens = [(token, count) for token, count in counter.items()]
        _tokens = sorted(_tokens, key=lambda x: -x[1])
        # 构建词列表
        _tokens = ['[PAD]', '[BOS]', '[EOS]', '[UNK]'] + [token for token, count in _tokens]
        # 创建token2id
        token2id = dict(zip(_tokens, range(len(_tokens))))
        # 根据token2id构建tokenizer
        self.tokenizer = Tokenizer(token2id)

    def seq_padding(self, data, max_len, padding=None):
        """
        把数据补全或截取至max_len
        :param data: 数据
        :param max_len: 序列最大长度
        :param padding: 需要补全的符号，default='[PAD]'=0
        :return: 补全之后的序列
        """
        # 获取需要填充的数据
        if padding is None:
            padding = self.tokenizer.token2id['[PAD]']
        self.padding = padding
        # 填充长度
        padding_len = max_len - len(data)
        # 填充
        if padding_len > 0:
            outputs = data + [padding] * padding_len
        # 截断
        else:
            outputs = data[:max_len]
        return outputs

    def __getitem__(self, item):
        x = np.array(self.seq_padding(self.tokenizer.encode(self.raw_data_x[item]), self.max_src_len))
        y = np.array(self.seq_padding(self.tokenizer.encode(self.raw_data_y[item]), self.max_target_len))
        if self.mode == 'train':
            return x, y
        else:
            lex = self.lexicalizations[item]
            muti_y = self.muti_data_y[''.join(self.raw_data_x[item])]
            return x, y, lex, muti_y

    def __len__(self):
        return len(self.ref)


if __name__ == '__main__':
    path = 'e2e_dataset/'
    train_dataset = E2EDataset(path + 'trainset.csv', mode='train')
    '''dev_dataset = E2EDataset(path + 'devset.csv', mode='dev',
                             field_tokenizer=train_dataset.field_tokenizer, tokenizer=train_dataset.tokenizer)
    test_dataset = E2EDataset(path + 'testset.csv', mode='test',
                              field_tokenizer=train_dataset.field_tokenizer, tokenizer=train_dataset.tokenizer)'''
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=256)
    for i, batch in enumerate(train_loader):
        source, target = batch
        if 162 <= i:
            print(i)
            print(f'source{source}, size{source.size()}')
            print(f'target{target}, size{target.size()}')
    # print(a[0])
    # print(a[1])
    # print(a[2])
    # print(a[3])
    '''token2word = {'[BOS]': 0, '[EOS]': 1, 'a': 2, '[PAD]': 3, 'b': 4, 'c': 5, '[UNK]':6}
    tokenizer = Tokenizer(token2word)
    print(tokenizer.encode(['a', 'a', 'b']))
    print(tokenizer.decode([0, 2, 2, 4, 1]))'''
