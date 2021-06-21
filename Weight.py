# -*- coding: UTF-8 -*-
# @CreateTime: 2021/6/18
# @LastUpdateTime: 2021/6/21
# @Author: Yingtong Hu

"""

Different weighting methods for SAO similarity calculation

"""

from BM25 import BM25
from FileProcess import FileProcess
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from pygraph.classes.graph import graph
from TFIDF import TFIDF


class Weight:
    km_labels = []
    sc_labels = []
    graph_degrees = []
    bm25_matrix = []
    bm25_mean = 0
    tfidfSys = None

    def __init__(self, patents, input_patent_SAO, vec, SAO):
        self.patents = patents
        self.input_patent_SAO = input_patent_SAO
        self.vec = vec
        self.SAO = SAO

    def set_up(self):
        self.km_labels = self.__KMeans_set_up()
        self.sc_labels = self.__SpectralClustering_set_up()
        self.graph_degrees = self.__Graph_set_up()
        self.bm25_matrix, self.bm25_mean = self.__bm25_set_up()
        self.__tfidf_set_up()

    def __bm25_set_up(self):
        values = self.input_patent_SAO.values()
        dataProcess = FileProcess("")
        SAOs = dataProcess.to_SAO(list(values)[0])
        test = []
        for SAO in SAOs:
            test.append(dataProcess.to_word(SAO))
        s = BM25(test)
        input_s = dataProcess.to_SAO(list(values)[1])
        matrix = []
        for i in input_s:
            matrix.append(s.similarity((dataProcess.to_word(i))))
        return matrix, np.mean(matrix)

    def __KMeans_set_up(self):
        cluster = KMeans(n_clusters=2).fit(self.vec)
        return cluster.labels_

    def __SpectralClustering_set_up(self):
        cluster = SpectralClustering(n_clusters=2, assign_labels='discretize').fit(self.vec)
        return cluster.labels_

    def __Graph_set_up(self):
        degrees = []
        for i in self.SAO:
            degrees.append(self.__SAOGraph(self.SAO[i][0], self.SAO[i][1], self.SAO[i][2]))
        return degrees

    @staticmethod
    def __SAOGraph(S, A, O):
        g = graph()
        g.add_nodes(S)

        for i in range(len(S)):
            if S[i] != '' and O[i] != '':
                if A[i] != '':
                    g.add_edge((S[i], A[i]))
                    g.add_edge((A[i], O[i]))
                    g.add_edge((S[i], O[i]))
            elif S[i] != '':
                if A[i] != '':
                    g.add_edge((S[i], A[i]))
            elif O[i] != '':
                if A[i] != '':
                    g.add_edge((A[i], O[i]))
        degrees = np.array([])
        for i in range(len(S)):
            degrees = np.append(degrees, len(g.neighbors(S[i])) + len(g.neighbors(A[i])) + len(g.neighbors(O[i])))

        return list(degrees / max(degrees))

    def __tfidf_set_up(self):
        self.tfidfSys = TFIDF(self.patents)

    def tfidf_(self, ind, SAOlist, ID, TF):
        return self.tfidfSys.tfidf(ind, SAOlist, ID, TF)
