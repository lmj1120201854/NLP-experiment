#-*- coding : utf-8-*-
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import jieba
import torch
import torchvision
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn

dots = ['，', '。', '！', '（', '）', '？', '；', '：', '《', '》', '\n', '!', '/', '.', ',', '"', "＂"]

words_vec = {}

model = KeyedVectors.load_word2vec_format("sgns.baidubaike.bigram-char/sgns.baidubaike.bigram-char", limit=100000)
# vector = model.most_similar(['男人'])
# wv = model["男人"]

class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(300, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        # 构建前向传播函数
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.softmax(self.fc2(x))
        return x

def train(x_train, y_train, x_test, y_test, model, loss_function, optimizer):
    for i in range(120):
        y_hat = model(x_train)
        loss = loss_function(y_hat, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.zero_grad()
    y_hat = model(x_test)
    y_hat = torch.max(y_hat, 1)[1].data.squeeze()
    score = torch.sum(y_hat == y_test).float() / y_test.shape[0]
    print(float(score))

def w2v_train(sentences, flag):
    for i in dots:
        sentences = sentences.replace(i, ' ')
    sentences = sentences.split()
    for sentence in sentences:
        seg_list = jieba.lcut(sentence)
        for words in seg_list:
            if words not in words_vec.keys():
                if flag == 1:
                    try:
                        words_vec[words] = [model[words], 1]
                    except KeyError:
                        words_vec[words] = [[0]*300, 1]
                else:
                    try:
                        words_vec[words] = [model[words], 0]
                    except KeyError:
                        words_vec[words] = [[0] * 300, 0]

def score():
    x_train = []
    y_train = []
    for words in words_vec.keys():
        x_train.append(words_vec[words][0])
        y_train.append(words_vec[words][1])
    x_test = x_train[:200]
    y_test = y_train[:200]
    x_train = x_train[200:]
    y_train = y_train[200:]
    x_testt = torch.tensor(x_test)
    y_testt = torch.tensor(y_test)
    x_traint = torch.tensor(x_train)
    y_traint = torch.tensor(y_train)

    model = net()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(x_traint, y_traint, x_testt, y_testt, model, loss_function, optimizer)

    clf = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='adam', learning_rate_init=0.001,
                        max_iter=1300)
    clf.fit(x_train, y_train)
    acc = clf.score(x_test, y_test)
    print(acc)

def main():
    for i in range(1000):
        f = open('negative/neg.{}.txt'.format(i), 'r', encoding='utf-8')
        t = f.read()
        w2v_train(t, 0)
    for i in range(1000):
        f = open('positive/pos.{}.txt'.format(i), 'r', encoding='utf-8')
        t = f.read()
        w2v_train(t, 1)
    score()

main()