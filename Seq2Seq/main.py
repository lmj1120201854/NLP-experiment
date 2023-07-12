import sys

import numpy as np
import torch

from config import Config
from dataset import E2EDataset
from prove_model import E2EModelProved
from model import E2EModel
from metric import BLEUScore

import tqdm
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# 配置
cfg = Config()
# 使用gpu
device = cfg.device
# 训练数据集加载
train_dataset = E2EDataset(path=cfg.train_dataset_path,
                           mode='train',
                           max_src_len=cfg.max_src_len,
                           max_target_len=cfg.max_target_len)
# 验证数据集加载
dev_dataset = E2EDataset(path=cfg.dev_dataset_path,
                         mode='dev',
                         max_src_len=cfg.max_src_len,
                         max_target_len=cfg.max_target_len,
                         field_tokenizer=train_dataset.field_tokenizer,
                         tokenizer=train_dataset.tokenizer)

# 定义数据加载器
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=cfg.batch_size)
dev_loader = DataLoader(dataset=dev_dataset,
                        batch_size=1)
# 定义模型
model = E2EModelProved(src_vocab_size=train_dataset.tokenizer.vocab_size,
                       target_vocab_size=train_dataset.tokenizer.vocab_size,
                       cfg=cfg).to(device)

# 损失函数
weight = torch.ones(train_dataset.tokenizer.vocab_size)
weight[cfg.PAD_ID] = 0
loss_function = nn.NLLLoss(weight=weight, size_average=True).to(device)

# 优化器
optimizer = optim.SGD(params=model.parameters(), lr=cfg.lr)

# 学习率随着迭代次数减少 -> 收敛更快
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# 声明分数计算器
scorer = BLEUScore(max_ngram=4)

# 记录模型保存的时机，达到最大bleu得分时保存模型到my_model.pt
best_bleu = 0
loss_list = []
bleu_list = []
learning_rate_list = []


def train(model, data_iter, iter):
    print_loss = 0.0
    model.train()
    with tqdm.tqdm(total=len(data_iter), desc='epoch{} [train]'.format(iter), file=sys.stdout) as t:
        for i, batch in enumerate(data_iter):
            if i >= 163:
                t.set_postfix(loss=print_loss / (i + 1), lr=scheduler.get_last_lr()[0])
                t.update(1)
                continue
            source, target = batch
            source = source.to(device).transpose(0, 1)
            target = target.to(device).transpose(0, 1)
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            logits = model((source, target))
            vocab_size = logits.size()[-1]
            logits = logits.contiguous().view(-1, vocab_size)
            targets = target.contiguous().view(-1, 1).squeeze(1)
            loss = loss_function(logits, targets.long())
            print_loss += loss.data.item()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 更新进度条
            t.set_postfix(loss=print_loss / (i + 1), lr=scheduler.get_last_lr()[0])
            t.update(1)
        # 记录loss、学习率的变化，更新学习率
        loss_list.append(print_loss / len(data_iter))
        learning_rate_list.append(scheduler.get_last_lr()[0])
        scheduler.step()

        t.close()
        # 每val_num轮进行一次验证
        if epoch % cfg.val_num == 0:
            evaluate(model, dev_dataset, epoch)


def evaluate(model, dev_iter, iter):
    global best_bleu
    model.eval()
    bleu = 0.0
    total_num = 0
    # 将分数计算器重置
    scorer.reset()
    with torch.no_grad():
        with tqdm.tqdm(total=len(dev_iter), desc='[dev]'.format(iter), file=sys.stdout) as t:
            for data in dev_iter:
                src, tgt, lex, muti_tgt = data
                src = torch.as_tensor(src[:, np.newaxis]).to(device)
                sentence, attention = model.predict(src)
                # decode
                sentence = train_dataset.tokenizer.decode(sentence).replace('xname', lex[0]).replace('xnear', lex[1])
                scorer.append(sentence, muti_tgt)
                t.update(1)
            t.close()
    bleu = scorer.score()
    bleu_list.append(bleu)
    print('BLEU SCORE: {:.4f}'.format(bleu))
    if bleu > best_bleu:
        best_bleu = bleu
        torch.save(model, 'my_best_model.pkl')
        draw_carve('valid_bleu', 'valid_bleu.png', epoch//cfg.val_num + 1, bleu_list)
    draw_carve('train_loss', 'train_loss.png', epoch + 1, loss_list)
    draw_carve('train_lr', 'train_lr.png', epoch + 1, learning_rate_list)


def draw_carve(title, save_path, x, y):
    plt.clf()
    plt.title(title)
    plt.plot(range(x), y)
    plt.savefig(save_path)


# 开始训练
for epoch in range(cfg.n_epoch):
    train(model, train_loader, epoch)

