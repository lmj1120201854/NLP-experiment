"""
模型入口，程序执行的开始，在该文件中配置必要的训练步骤
"""


from Exp3_Config import Training_Config
import Exp3_DataSet
from Exp3_DataSet import TextDataSet, TestDataSet
from torch.utils.data import DataLoader
from Exp3_Model import TextCNN_Model, BiLSTM_model
import torch
import time

config = Training_Config()
if config.cuda:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train(model, loader):
    sum_true = 0
    sum_loss = 0.0
    for index, data in enumerate(loader):
        # print(data['text'])
        y_hat = model(data['text'].to(device), data['pos1'].to(device), data['pos2'].to(device))
        y_label = data['relation'].to(device)

        loss = loss_function(y_hat, y_label)

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数

        y_hat = torch.tensor([torch.argmax(_) for _ in y_hat]).to(device)
        sum_true += torch.sum(y_hat == y_label).float()
        sum_loss += loss.item()

    train_acc = sum_true / (loader.__len__() * config.batch_size)
    train_loss = sum_loss / loader.__len__()

    print(f"train loss: {train_loss:.4f}, train accuracy: {train_acc * 100:.2f}")


def validation(model, loader):
    sum_true = 0
    sum_loss = 0.0
    with torch.no_grad():
        for index, data in enumerate(loader):
            y_hat = model(data['text'].to(device), data['pos1'].to(device), data['pos2'].to(device))
            y_label = data['relation'].to(device)

            loss = loss_function(y_hat, y_label)

            y_hat = torch.tensor([torch.argmax(_) for _ in y_hat]).to(device)
            sum_true += torch.sum(y_hat == y_label).float()
            sum_loss += loss.item()

    val_acc = sum_true / (loader.__len__() * config.batch_size)
    val_loss = sum_loss / loader.__len__()

    print(f"val loss: {val_loss:.4f}, val accuracy: {val_acc * 100:.2f}")


def predict(model, loader):
    pre = []
    with torch.no_grad():
        for index, data in enumerate(loader):
            y_hat = model(data['text'].to(device), data['pos1'].to(device), data['pos2'].to(device))
            temp = [int(torch.argmax(_)) for _ in y_hat]
            pre = pre + temp
    f = open('output.txt', 'w')
    for line in pre:
        f.write(str(line) + '\n')
    f.close()


if __name__ == "__main__":
    rel2id, id2rel, word2token, token2word = Exp3_DataSet.trans_json()
    # 训练集验证集
    train_dataset = TextDataSet(filepath="./data/data_train.txt", config=config,
                                rel2id=rel2id, id2rel=id2rel, word2token=word2token, token2word=token2word)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size)

    val_dataset = TextDataSet(filepath="./data/data_val.txt", config=config,
                              rel2id=rel2id, id2rel=id2rel, word2token=word2token, token2word=token2word)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size)

    # 测试集数据集和加载器
    test_dataset = TestDataSet("./data/test_exp3.txt", config=config,
                               word2token=word2token, token2word=token2word)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size)

    # 初始化模型对象
    Text_Model = TextCNN_Model(configs=config).to(device)
    # Text_Model = BiLSTM_model(configs=config).to(device)

    # 损失函数设置
    loss_function = torch.nn.CrossEntropyLoss()  # torch.nn中的损失函数进行挑选，并进行参数设置
    # 优化器设置
    optimizer = torch.optim.Adam(params=Text_Model.parameters(),
                                 lr=config.lr, weight_decay=5e-4)  # torch.optim中的优化器进行挑选，并进行参数设置

    # 训练和验证
    for i in range(config.epoch):
        train(Text_Model, loader=train_loader)
        if i % config.num_val == 0:
            validation(Text_Model, loader=val_loader)
        if i % 10 == 0:
            torch.save(Text_Model, 'model.pt')
    torch.save(Text_Model, 'model.pt')
    # 预测（测试）
    Text_Model = torch.load('model.pt')
    predict(Text_Model, test_loader)













