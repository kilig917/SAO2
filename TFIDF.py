# -*- coding: UTF-8 -*-
# @CreateTime: 2021/6/18
# @LastUpdateTime: 2021/6/18
# @Author: Yingtong Hu

"""

TFIDF weighting method

"""

import re
import numpy as np


class TFIDF(object):
    def __init__(self, patents):
        self.patents = patents
        self.IDF_count, self.doc_num = self.IDF()

    def IDF(self):
        word_IDF = {}  # word - [patent]
        patent_ID = []
        for input_patent_SAO in self.patents:
            for i in input_patent_SAO:
                if i not in patent_ID:
                    patent_ID.append(i)
                text = re.split(';', input_patent_SAO[i])
                for n, j in enumerate(text):
                    if j[:3] == '^_^':
                        j = j[3:]
                    # split SAO
                    SAO = j.split(', ')
                    word = (SAO[0][1:], SAO[1], SAO[2][:-1])
                    # IDF
                    if word in word_IDF:
                        if i not in word_IDF[word]:
                            word_IDF[word].append(i)
                    else:
                        word_IDF[word] = [i]
        return word_IDF, len(patent_ID)

    def tfidf(self, ind, SAOlist, ID, TF):
        if SAOlist[0][ind] == ' ':
            word = (SAOlist[1][ind], SAOlist[2][ind])
        else:
            word = (SAOlist[0][ind], SAOlist[1][ind], SAOlist[2][ind])
        tf = TF[ID][word] / len(TF[ID])
        idf = np.log(self.doc_num / (len(self.IDF_count[word]) + 1))
        if idf == 0.0:
            idf = 0.0001
        tfidf = tf * idf
        return tfidf
