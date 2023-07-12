import jieba

dots = ['，', '。', '！', '（', '）', '？', '；', '：', '《', '》', '\n', '!', '/', '.', ',', '"', "＂"]
poswords = {} # 存放积极词典的词汇
negwords = {} # 存放消极词典的词汇
poswords_num = 0
negwords_num = 0
words_pr_pos = {} # P(X=x|Y=pos)
words_pr_neg = {} # P(X=x|Y=neg)
total_sentence = 4000
pos_pr = 1/2
neg_pr = 1/2 # P(Y=pos) = P(Y=neg) = 1/2
nx_test = [[]]
px_test = [[]]
# 导入情感词典并预处理
def input_data():
    # 导入消极词典
    num1 = 0
    num2 = 0
    for i in range(1000):
        f = open('negative/neg.{}.txt'.format(i), 'r', encoding='utf-8')
        t = f.read()
        for j in dots:
            t = t.replace(j, ' ')
        t = t.split()
        for sentence in t:
            seg_list = jieba.cut(sentence)
            for words in seg_list:
                num1 = num1 + 1
                if words not in negwords.keys():
                    negwords[words] = 1
                else:
                    negwords[words] = negwords[words] + 1
        if i > 800: #导入测试集
            for sentence in t:
                seg_list = jieba.cut(sentence)
                for words in seg_list:
                    if len(nx_test) < i - 800:
                        nx_test.append([words])
                    else:
                        nx_test[i - 801].append(words)

    # 导入积极词典
    for i in range(1000):
        f = open('positive/pos.{}.txt'.format(i), 'r', encoding='utf-8')
        t = f.read()
        for j in dots:
            t = t.replace(j, ' ')
        t = t.split()
        for sentence in t:
            seg_list = jieba.cut(sentence)
            for words in seg_list:
                num2 = num2 + 1
                if words not in poswords.keys():
                    poswords[words] = 1
                else:
                    poswords[words] = poswords[words] + 1
        if i > 800: #导入测试集
            for sentence in t:
                seg_list = jieba.cut(sentence)
                for words in seg_list:
                    if len(px_test) < i - 800:
                        px_test.append([words])
                    else:
                        px_test[i - 801].append(words)
    return num2, num1

# 每个词的概率计算
def pr_compute(word):
    if word in words_pr_pos:
        words_pr_pos[word] = words_pr_pos[word] ** 2
    else:
        if word in poswords.keys():
            words_pr_pos[word] = poswords[word] / poswords_num
        else:
            words_pr_pos[word] = 1 / poswords_num
    if word in words_pr_neg:
        words_pr_neg[word] = words_pr_neg[word] ** 2
    else:
        if word in negwords.keys():
            words_pr_neg[word] = negwords[word] / negwords_num
        else:
            words_pr_neg[word] = 1 / negwords_num

# 句子正负性判断
def score():
    pospr = 1
    negpr = 1
    flag = 0
    for i in words_pr_pos.values():
        # 由于语料较少是否考虑概率为0？
        pospr = pospr * i
    for i in words_pr_neg.values():
        negpr = negpr * i
    if pospr > negpr:
        flag = 1
    return flag

def main():
    '''wordlist = []
    t = "标准的商务酒店，最大的优点就是位置好。房间比较干净，设施还行，但有些细节还需要改进。".encode(encoding='utf-8')
    t = t.decode(encoding='utf-8')
    for j in dots:
        t = t.replace(j, ' ')
    t = t.split()
    for shortsen in t:
        wordlist.append(jieba.cut(shortsen))
    for i in wordlist:
        for word in i:
            pr_compute(word)
    res = score()
    if res == 0:
        print("消极")
    else:
        print("积极")'''
    all_sentence = 400
    true_sentence = 0
    for words_list in nx_test:
        for word in words_list:
            pr_compute(word)
        if score() == 0:
            true_sentence = true_sentence + 1
        words_pr_neg.clear()
        words_pr_pos.clear()
    for words_list in px_test:
        for word in words_list:
            pr_compute(word)
        if score() == 1:
            true_sentence = true_sentence + 1
        words_pr_neg.clear()
        words_pr_pos.clear()

    print(true_sentence)
    print(true_sentence / all_sentence)

poswords_num, negwords_num = input_data()
main()
