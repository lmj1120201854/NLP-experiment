import torch
from torch import nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from config import Config
from dataset import E2EDataset
from model import E2EModel
from prove_model import E2EModelProved

import warnings
warnings.filterwarnings("ignore")

cfg = Config()
device = cfg.device

model = torch.load('my_best_model.pkl').to(device)
dataset = E2EDataset(cfg.train_dataset_path, mode='train')
dataset.mode = 'dev'

src, tgt, lex, _ = dataset[0]
src = torch.as_tensor(src[np.newaxis, :]).to(device).transpose(0, 1)
sentence, attention = model.predict(src)

src_txt = list(map(lambda x: dataset.tokenizer.id2token[x], src.flatten().cpu().numpy().tolist()[:10]))
for i in range(len(src_txt)):
    if src_txt[i] == 'xname':
        src_txt[i] = lex[0]
    elif src_txt[i] == 'xnear':
        src_txt[i] = lex[1]
sentence_txt = list(map(lambda x: dataset.tokenizer.id2token[x], sentence))
for i in range(len(src_txt)):
    if sentence_txt[i] == 'xname':
        sentence_txt[i] = lex[0]
    elif sentence_txt[i] == 'xnear':
        sentence_txt[i] = lex[1]

# 绘制热力图
ax = sns.heatmap(np.array(attention)[:, :10] * 100, cmap='YlGnBu')
# 设置坐标轴
plt.yticks([i + 0.5 for i in range(len(sentence_txt))], labels=sentence_txt, rotation=360, fontsize=4)
plt.xticks([i + 0.5 for i in range(len(src_txt))], labels=src_txt, rotation=45, fontsize=6)
plt.show()

