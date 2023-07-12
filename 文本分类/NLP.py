#-*- coding : utf-8-*-

import jieba

#导入停用词
stopwords = set()
fr = open('vocabulary/stopwords_utf8.txt', 'r', encoding='utf-8')
for word in fr:
    stopwords.add(word.strip())

#导入程度副词
import openpyxl
degreewords = []
degree = []
data = openpyxl.load_workbook("vocabulary/degree_dict.xlsx")
ws = data.active
for line in ws:
    degreewords.append(line[0].value)
    degree.append(line[1].value)
del degree[0]
del degreewords[0]

#导入正负词汇
negwords = []
poswords = []
fr = open('vocabulary/full_neg_dict_sougou.txt', 'r', encoding='utf-8')
for word in fr:
    negwords.append(word.strip())
fr = open('vocabulary/full_pos_dict_sougou.txt', 'r', encoding='utf-8')
for word in fr:
    poswords.append(word.strip())

#分词
def seg_words(sentence):
    seg_list = jieba.cut(sentence)
    seg_result = []
    for i in seg_list:
        if i in stopwords:
            continue
        elif i==' ':
            continue
        else:
            seg_result.append(i)
    return seg_result

#打分
def score(seg_result):
    w = 1
    sco = 0
    for i in seg_result:
        if i in degreewords:
            index = degreewords.index(i)
            w = w * degree[index]
        elif i in negwords:
            sco = sco - 1
        elif i in poswords:
            sco = sco + 1
    return sco * w

'''res = seg_words("酒店位置离市中心不远,交通很方便.大堂装饰带些欧式风格,空间虽不像有些酒店那么大，但很温馨。")
print(res)
s = score((res))
print(s)
if s >= 0:
    print("积极")
else:
    print("消极")'''

dots = ['，', '。', '！', '（', '）', '？', '；', '：', '《', '》', '\n', '!', '/', '.', ',', '"', "＂"]
all_sentence = 0
true_sentence = 0
for i in range(1000):
    f = open('negative/neg.{}.txt'.format(i), 'r', encoding='utf-8')
    t = f.read()
    for j in dots:
        t = t.replace(j, ' ')
    t = t.split()
    for sentence in t:
        all_sentence += 1
        seg_list = jieba.cut(sentence)
        if score(seg_list) < 0:
            true_sentence += 1
        if score(seg_list) == 0:
            true_sentence += 1/2
for i in range(1000):
    f = open('positive/pos.{}.txt'.format(i), 'r', encoding='utf-8')
    t = f.read()
    for j in dots:
        t = t.replace(j, ' ')
    t = t.split()
    for sentence in t:
        all_sentence += 1
        seg_list = jieba.cut(sentence)
        if score(seg_list) > 0:
            true_sentence += 1
        if score(seg_list) == 0:
            true_sentence += 1/2

print(true_sentence / all_sentence)