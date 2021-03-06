# !/usr/bin/env python 
# -*- coding: UTF-8 -*- 
# @Time: 2020/7/20 21:50 
# @Author: Zhang Cong
# @LastUpdateTime: 2021/6/18
# @Updater: Yingtong Hu

import math
import jieba
import numpy as np


class BM25(object):
    def __init__(self, docs):
        self.N = len(docs)  # 文本数量
        self.avgdl = sum([len(doc) for doc in docs]) * 1.0 / self.N  # 文本平均长度
        self.docs = docs
        self.f = []  # 每篇文档中每个词的出现次数
        self.df = {}  # 每个词及出现了该词的文档数量
        self.idf = {}  # 每个词的IDF值
        self.k1 = 1.5  # 调节参数K1
        self.b = 0.75  # 调节参数b
        self.init()

    def init(self):
        """
        计算文档集每篇文档中每个词的出现次数、每个词及出现了该词的文档数量、每个词的IDF值
        :return:
        """
        for doc in self.docs:
            tmp = {}
            # 统计当前文档中每个词的出现次数
            for word in doc:
                tmp[word] = tmp.get(word, 0) + 1
            self.f.append(tmp)  # 加入到全局记录中

            # 统计出现了当前词汇的文档数量
            for k in tmp.keys():
                self.df[k] = self.df.get(k, 0) + 1

        # 计算IDF值
        for k, v in self.df.items():
            self.idf[k] = math.log(self.N - v + 0.5) - math.log(v + 0.5)

    def get_score(self, query, index):
        '''
        计算输入的query和doc的相似度分数score
        :param doc: 输入的query
        :param index: 文档集中的文档索引
        :return:
        '''
        score = 0
        for word in query:
            # 如果是未登录词，则跳过
            if word not in self.f[index]:
                continue
            dl = len(self.docs[index])  # 当前文档长度
            # 计算相似度分数 IDF*R(q, d) 求和
            score += (self.idf[word] * self.f[index][word] * (self.k1 + 1)
                      / (self.f[index][word] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)))
        return score

    def similarity(self, query):
        '''
        输入query对文档集进行检索
        :param doc: 分词后的query list
        :return:
        '''
        scores = []
        for index in range(self.N):
            score = self.get_score(query, index)
            scores.append(score)
        return scores
