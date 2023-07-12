import math
from collections import defaultdict


class BLEUScore:
    TINY = 1e-15
    SMALL = 1e-9

    def __init__(self, max_ngram=4, case_sensitive=False):
        self.max_ngram = max_ngram  # 最大 n-gram
        self.case_sensitive = case_sensitive
        self.ref_len = 0
        self.cand_lens = [0] * self.max_ngram
        self.hits = [0] * self.max_ngram

    def reset(self):
        self.ref_len = 0
        self.cand_lens = [0] * self.max_ngram
        self.hits = [0] * self.max_ngram

    def append(self, pred_sent, ref_sents):
        pred_sent = pred_sent
        ref_sents = [ref_sent for ref_sent in ref_sents]
        for i in range(self.max_ngram):
            # 计算每个 gram 的命中次数
            self.hits[i] += self.compute_hits(i + 1, pred_sent, ref_sents)
            # 计算每个 gram 的预测长度
            self.cand_lens[i] += len(pred_sent) - i
        # 选择长度最相近的参考文本
        closest_ref = min(ref_sents, key=lambda ref_sent: (abs(len(ref_sent) - len(pred_sent)), len(ref_sent)))
        # 记录参考文本长度
        self.ref_len += len(closest_ref)

    def compute_hits(self, n, pred_sent, ref_sents):
        merged_ref_ngrams = self.get_ngram_counts(n, ref_sents)
        pred_ngrams = self.get_ngram_counts(n, [pred_sent])
        hits = 0
        for ngram, cnt in pred_ngrams.items():
            hits += min(merged_ref_ngrams.get(ngram, 0), cnt)
        return hits

    def get_ngram_counts(self, n, sents):
        merged_ngrams = {}
        # 按 gram 数聚合句子
        for sent in sents:
            ngrams = defaultdict(int)
            if not self.case_sensitive:
                ngrams_list = list(zip(*[[tok.lower() for tok in sent[i:]] for i in range(n)]))
            else:
                ngrams_list = list(zip(*[sent[i:] for i in range(n)]))
            for ngram in ngrams_list:
                ngrams[ngram] += 1
            for ngram, cnt in ngrams.items():
                merged_ngrams[ngram] = max((merged_ngrams.get(ngram, 0),
                                            cnt))
        return merged_ngrams

    def score(self):
        bp = 1.0
        # c <= r : BP=e^(1-r/c)
        # c > r : BP=1.0
        if self.cand_lens[0] <= self.ref_len:
            bp = math.exp(1.0 - self.ref_len / (float(self.cand_lens[0])
                                                if self.cand_lens[0] else 1e-5))
        prec_log_sum = 0.0
        for n_hits, n_len in zip(self.hits, self.cand_lens):
            n_hits = max(float(n_hits), self.TINY)

            n_len = max(float(n_len), self.SMALL)
            # 计算∑logPn=∑log(n_hits/n_len)
            prec_log_sum += math.log(n_hits / n_len)
        return bp * math.exp((1.0 / self.max_ngram) * prec_log_sum)


if __name__ == '__main__':

     scorer = BLEUScore(max_ngram=2)
     sentence = 'the cat sat on the mat '
     target = ['the cat is on the mat ']
     scorer.append(sentence, target)
     print(scorer.score())

