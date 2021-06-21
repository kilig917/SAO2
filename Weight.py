# -*- coding: UTF-8 -*-
# @CreateTime: 2021/6/18
# @LastUpdateTime: 2021/6/18
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


class Weight(object):
    def __init__(self, patents):
        self.tfidfSys = TFIDF(patents)

    def bm25(self, input_patent_SAO):
        values = input_patent_SAO.values()
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

    def KM(self, vec):
        cluster = KMeans(n_clusters=2).fit(vec)
        return cluster.labels_

    def SC(self, vec):
        cluster = SpectralClustering(n_clusters=2, assign_labels='discretize').fit(vec)
        return cluster.labels_

    def SAOGraph(self, S, A, O):
        g = graph()
        g.add_nodes(S)
        g.add_nodes(A)
        g.add_nodes(O)
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
        degrees = []
        for i in range(len(S)):
            degrees.append(len(g.neighbors(S[i])) + len(g.neighbors(A[i])) + len(g.neighbors(O[i])))
        degrees = np.array(degrees)
        degrees = list(degrees / max(degrees))

        return degrees

    def TfIdf(self, ind, SAOlist, ID, TF):
        return self.tfidfSys.tfidf(ind, SAOlist, ID, TF)