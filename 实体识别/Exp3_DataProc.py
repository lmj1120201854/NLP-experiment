"""
这个文件中可以添加数据预处理相关函数和类等
如词汇表生成，Word转ID（即词转下标）等
此文件为非必要部分，可以将该文件中的内容拆分进其他部分
"""
import os
import json


class VocabCatch(object):
    def __init__(self):
        self.word2tkn = {'': 0}
        self.tkn2word = [' ']  # 此处初始化空格的作用是方便padding，padding填充的0等同于文本填充的空格

    def tokenize(self, path, test_mod=False):
        if test_mod:  # test数据集与其他的数据集不同，需要单独挑出
            lines = open(path, 'r', encoding='utf-8').readlines()
            for line in lines:
                text = line.split('\t')[2][:-1]
                for word in text:
                    self.add_word(word)
        else:
            lines = open(path, 'r', encoding='utf-8').readlines()
            for line in lines:
                text = line.split('\t')[3][:-1]
                for word in text:
                    self.add_word(word)

    def add_word(self, word):  # 当遇到此表中没有的字符的时候，需要扩充
        if word not in self.word2tkn:
            self.tkn2word.append(word)
            self.word2tkn[word] = len(self.tkn2word) - 1
        return self.word2tkn[word]


if __name__ == '__main__':
    print("数据预处理开始......")
    vocab = VocabCatch()
    vocab.tokenize('data/data_train.txt')
    vocab.tokenize('data/data_val.txt')
    vocab.tokenize('data/test_exp3.txt', True)
    json_list = [vocab.tkn2word, vocab.word2tkn]
    with open('data/vocab.json', mode='w', encoding='utf-8') as f:
        json.dump(json_list, f)  # 将字典和解码列表写入json文件中
    with open('data/vocab.json', mode='r', encoding='utf-8') as f:
        dicts = json.load(f)
        # 将多个字典从json文件中读出来
        for i in dicts:
            print(len(i))
    print("数据预处理完毕！")
