# -*- coding: UTF-8 -*-
# @CreateTime: 2021/6/18
# @LastUpdateTime: 2021/6/23
# @Author: Yingtong Hu

"""

Different weighting methods for SAO similarity calculation

"""

import numpy as np
import re
from BM25 import BM25
from sklearn.cluster import KMeans, SpectralClustering
from pygraph.classes.graph import graph
import SAO


class Weight:
    """
    Combining different weighting methods: BM25, KMeans, Spectral Clustering, Graph and TFIDF
    """
    km_labels = []
    sc_labels = []
    graph_degrees = []
    bm25_matrix = []
    bm25_mean = 0
    IDF_count = {}
    doc_num = 0

    def __init__(self, patents, input_patent_SAO, vec, SAO, methods):
        self.patents = patents
        self.input_patent_SAO = input_patent_SAO
        self.vec = vec
        self.SAO = SAO
        self.methods = methods

    def set_up(self):
        """
        getting weight calculation required variables ready
        :return: None
        """
        self.km_labels = self.__KMeans_set_up()
        self.sc_labels = self.__SpectralClustering_set_up()
        self.graph_degrees = self.__Graph_set_up()
        self.bm25_matrix, self.bm25_mean = self.__bm25_set_up()
        self.IDF_count, self.doc_num = self.__tfidf_set_up()

    def __bm25_set_up(self):
        """
        set up bm25
        :return: matrix of bm25 values; mean of all bm25 values
        """
        values = self.input_patent_SAO.values()
        SAOs = self.__to_SAO(list(values)[0])
        test = []
        for SAOComb in SAOs:
            test.append(self.__to_word(SAOComb))
        s = BM25(test)
        input_s = self.__to_SAO(list(values)[1])
        matrix = []
        for i in input_s:
            matrix.append(s.similarity((self.__to_word(i))))
        return matrix, np.mean(matrix)

    @staticmethod
    def __to_SAO(sentence):
        """
        For BM25 calculation
        split string of all SAOs to array of SAOs
        eg. "^_^(aaa, bbb, ccc); (aaa, bbb, ccc)" --> ["(aaa, bbb, ccc)", "(aaa, bbb, ccc)"]
        """
        sentence = sentence.replace('^_^', '').replace(';', '')
        SAOs = sentence.split(')')
        if '' in SAOs:
            SAOs.remove('')
        for i in range(len(SAOs)):
            SAOs[i] += ')'
        return SAOs

    @staticmethod
    def __to_word(SAOComb):
        """
        For BM25 calculation
        split SAOs to single word combination
        eg. (aaa, bbb, ccc) --> [aaa, bbb, ccc]
        """
        word = SAOComb.split(', ')
        for i in range(len(word)):
            word[i] = word[i].replace('(', '').replace(')', '')
        if '' in word:
            word.remove('')
        return word

    def __KMeans_set_up(self):
        """
        set up KMeans
        :return: KMeans labels
        """
        cluster = KMeans(n_clusters=2).fit(self.vec)
        return cluster.labels_

    def __SpectralClustering_set_up(self):
        """
        set up Spectral Clustering
        :return: Spectral Clustering labels
        """
        cluster = SpectralClustering(n_clusters=2, assign_labels='discretize').fit(self.vec)
        return cluster.labels_

    def __Graph_set_up(self):
        """
        set up Graph
        :return: graph degrees
        """
        degrees = []
        for i in self.SAO:
            degrees.append(self.__SAOGraph(self.SAO[i][0], self.SAO[i][1], self.SAO[i][2]))
        return degrees

    @staticmethod
    def __SAOGraph(S, A, O):
        """
        create graph
        :param S: Subject
        :param A: Action
        :param O: Object
        :return: normalized degree
        """
        g = graph()
        g.add_nodes(S + A + O)
        for i in range(len(S)):
            if S[i] != '' and O[i] != '' and A[i] != '':
                g.add_edge((S[i], O[i]))
            if S[i] != '' and A[i] != '':
                g.add_edge((S[i], A[i]))
            if O[i] != '' and A[i] != '':
                g.add_edge((A[i], O[i]))
        degrees = np.array([])
        for i in range(len(S)):
            degrees = np.append(degrees, len(g.neighbors(S[i])) + len(g.neighbors(A[i])) + len(g.neighbors(O[i])))
        return list(degrees / max(degrees))

    def __tfidf_set_up(self):
        """
        set up TFIDF
        :return: each word's IDF value; total number of documents
        """
        word_IDF = {}  # word - [patent]
        patent_ID = []
        for input_patent_SAO in self.patents:
            for i in input_patent_SAO:
                if i not in patent_ID:
                    patent_ID.append(i)
                text = re.split(';', input_patent_SAO[i])
                for n, j in enumerate(text):
                    j = SAO.SAOLabel(j)[1]
                    s, a, o = SAO.splitSAO(j)
                    # IDF
                    if (s, a, o) in word_IDF:
                        if i not in word_IDF[(s, a, o)]:
                            word_IDF[(s, a, o)].append(i)
                    else:
                        word_IDF[(s, a, o)] = [i]
        return word_IDF, len(patent_ID)

    def bm25(self, SAO_index, similarity, similarity_w_weight):
        """
        calculate bm25 weight
        :param SAO_index: current SAO pair
        :param similarity: raw similarity
        :param similarity_w_weight: similarity with weight added
        :return: updated similarity_w_weight
        """
        if self.bm25_matrix[SAO_index[1]][SAO_index[0]] > self.bm25_mean:
            for index in range(len(self.methods)):
                if 'bm25' in similarity_w_weight[index].keys():
                    similarity_w_weight[index]['bm25'] += similarity[index][SAO_index]
                else:
                    similarity_w_weight[index]['bm25'] = similarity[index][SAO_index]
        return similarity_w_weight

    def KMeans(self, SAO_index, source, similarity, similarity_w_weight):
        """
        calculate KMeans weight
        :param SAO_index: current SAO pair
        :param source: source patent
        :param similarity: raw similarity
        :param similarity_w_weight: similarity with weight added
        :return: updated similarity_w_weight
        """
        km_weight = 1 if self.km_labels[SAO_index[0]] == self.km_labels[len(source[0]) + SAO_index[1]] else 0
        km_score = similarity
        if km_weight == 1:
            km_score = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        if 'km' in similarity_w_weight[0].keys():
            for index in range(len(self.methods)):
                similarity_w_weight[index]['km'] += km_score[index]
        else:
            for index in range(len(self.methods)):
                similarity_w_weight[index]['km'] = km_score[index]
        return similarity_w_weight

    def SpectralClustering(self, SAO_index, source, similarity, similarity_w_weight):
        """
        calculate Spectral Clustering weight
        :param SAO_index: current SAO pair
        :param source: source patent
        :param similarity: raw similarity
        :param similarity_w_weight: similarity with weight added
        :return: updated similarity_w_weight
        """
        sc_weight = 1 if self.sc_labels[SAO_index[0]] == self.sc_labels[len(source[0]) + SAO_index[1]] else 0
        sc_score = similarity
        if sc_weight == 1:
            sc_score = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        if 'sc' in similarity_w_weight[0].keys():
            for index in range(len(self.methods)):
                similarity_w_weight[index]['sc'] += sc_score[index]
        else:
            for index in range(len(self.methods)):
                similarity_w_weight[index]['sc'] = sc_score[index]
        return similarity_w_weight

    def Graph(self, SAO_index, similarity, similarity_w_weight):
        """
        calculate Graph weight
        :param SAO_index: current SAO pair
        :param similarity: raw similarity
        :param similarity_w_weight: similarity with weight added
        :return: updated similarity_w_weight
        """
        d1 = self.graph_degrees[0][SAO_index[0]]
        d2 = self.graph_degrees[1][SAO_index[1]]
        if 'graph' in similarity_w_weight[0].keys():
            for index in range(len(self.methods)):
                similarity_w_weight[index]['graph'] += similarity[
                                             index] * (d1 + d2) / 2
        else:
            for index in range(len(self.methods)):
                similarity_w_weight[index]['graph'] = similarity[
                                            index] * (d1 + d2) / 2
        return similarity_w_weight

    def tfidf(self, SAO_index, source, target, TF_count, similarity, similarity_w_weight):
        """
        calculate TFIDF weight
        :param SAO_index: current SAO pair
        :param source: source patent SAO list
        :param target: target patent SAO list
        :param TF_count: total TF count
        :param similarity: raw similarity
        :param similarity_w_weight: similarity with weight added
        :return: updated similarity_w_weight
        """
        w1 = self.__get_tfidf_val(SAO_index[0], source, SAO.sourceID, TF_count)
        w2 = self.__get_tfidf_val(SAO_index[1], target, SAO.targetID, TF_count)
        if 'tfidf' in similarity_w_weight[0].keys():
            for index in range(len(self.methods)):
                similarity_w_weight[index]['tfidf'] += similarity[index] * (w1 * w2)
        else:
            for index in range(len(self.methods)):
                similarity_w_weight[index]['tfidf'] = similarity[index] * (w1 * w2)
        return similarity_w_weight

    def __get_tfidf_val(self, ind, SAOList, ID, TF_count):
        """
        get TFIDF value
        :param ind: SAO index
        :param SAOList: patent's SAO list
        :param ID: patent ID
        :param TF_count: total TF count
        :return: TFIDF value for this SAO
        """
        if SAOList[0][ind] == ' ':
            word = (SAOList[1][ind], SAOList[2][ind])
        else:
            word = (SAOList[0][ind], SAOList[1][ind], SAOList[2][ind])
        tf = TF_count[ID][word] / len(TF_count[ID])
        idf = np.log(self.doc_num / (len(self.IDF_count[word]) + 1))
        if idf == 0.0:
            idf = 0.0001
        tfidf_val = tf * idf
        return tfidf_val
