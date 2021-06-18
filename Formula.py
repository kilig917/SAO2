# -*- coding: UTF-8 -*-
# @Time: 2021/6/18
# @Author: Yingtong Hu

"""

Calculation Formulas for different methods
including: Dice coefficient, inclusion index, Jaccard coefficient, Euclidean distance, Pearson coefficient,
            Spearman coefficient, Arcosine distance and different methods of concept hierarchy

"""

import numpy as np
from pandas.core.frame import DataFrame
from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic


class Formula(object):
    def __init__(self, SAO1, SAO2, type):
        if type == 'w':
            self.set1 = set(SAO1.split(' '))
            self.set2 = set(SAO2.split(' '))
        elif type == 'v':
            self.vector1 = np.array(SAO1)
            self.vector2 = np.array(SAO2)
        else:
            self.brown_ic = wordnet_ic.ic('ic-brown.dat')
            self.semcor_ic = wordnet_ic.ic('ic-semcor.dat')
            self.errorpos = ['a', 'r', 's']

    def dice(self):
        """dice coefficient 2nt/(na + nb)."""
        overlapLen = len(self.set1 & self.set2)
        similarity = overlapLen * 2.0 / (len(self.set1) + len(self.set2))
        return similarity

    def inclusionIndex(self):
        """inclusion index nt/min(na, nb)."""
        overlap = self.set1 & self.set2
        similarity = len(overlap) / min(len(self.set1), len(self.set2))
        return similarity

    def jaccard(self):
        """Jaccard coefficient nt/n(aUb)."""
        similarity = len(set.intersection(self.set1, self.set2)) / len(set.union(self.set1, self.set2))
        return similarity

    def euclidean(self):
        """Euclidean Distance """
        d = np.sqrt(np.sum(np.square(self.vector1 - self.vector2)))
        return 1 / (1 + d)

    def pearson(self):
        """Pearson Coefficient"""
        data = DataFrame({'x': self.vector1, 'y': self.vector2})
        similarity = abs(data.corr()['x']['y'])
        if np.isnan(similarity):
            return 1.0
        return similarity

    def spearman(self):
        """Spearman Coefficient"""
        data = DataFrame({'x': self.vector1, 'y': self.vector2})
        similarity =  abs(data.corr(method='spearman')['x']['y'])
        if np.isnan(similarity):
            return 1.0
        return similarity

    def arcosine(self, v1, v2):
        """Arcosine Distance"""
        a = np.sqrt(sum([np.square(x) for x in v1]))
        b = np.sqrt(sum([np.square(x) for x in v2]))
        if a == 0.0 or b == 0.0:
            return 0.0
        d = np.dot(v1, v2) / (a * b)
        return 0.5 * d + 0.5

    def hierarchy(self, word1, word2):
        """Concept Hierarchy top-level function"""
        temp_word1 = word1.split(' ')
        temp_word2 = word2.split(' ')
        d = [0.0, 0.0, 0.0]
        for t1 in temp_word1:
            for t2 in temp_word2:
                result = self.hierarchyCalculation(t1, t2)
                for i in range(3):
                    d[i] += result[i]
        return d

    def hierarchyCalculation(self, w1, w2):
        """Concept Hierarchy Calculation"""
        if w1 == w2:
            return [1.0, 1.0, 1.0]
        synsets1, synsets2 = wordnet.synsets(w1), wordnet.synsets(w2)
        if not synsets1 or not synsets2:
            return [0.0, 0.0, 0.0]
        maxx = [0.0, 0.0, 0.0]
        for s1 in synsets1:
            for s2 in synsets2:
                # maxx[2] = self.wu(s1, s2, maxx[2])
                if s1._pos == s2._pos and s1._pos not in self.errorpos and s2._pos not in self.errorpos:
                    # maxx[1] = self.leacock(s1, s2, maxx[1])
                    maxx = [self.lin(s1, s2, maxx[0]), self.resnik(s1, s2, maxx[1]), self.jiang(s1, s2, maxx[2])]
        return maxx

    def wu(self, s1, s2, val):
        """Concept Hierarchy Calculation"""
        sim_wu = s1.wup_similarity(s2)
        if sim_wu is not None and sim_wu > val:
            return sim_wu
        return -1

    def lin(self, s1, s2, val):
        sim_lin = s1.lin_similarity(s2, self.semcor_ic)
        if sim_lin is not None and sim_lin > val:
            return sim_lin
        return -1

    def leacock(self, s1, s2, val):
        sim1_leacock = s1.lch_similarity(s1)
        sim2_leacock = s2.lch_similarity(s2)
        sim_leacock = s1.lch_similarity(s2) / max(sim1_leacock, sim2_leacock)
        if sim_leacock > val:
            return sim_leacock
        return val

    def resnik(self, s1, s2, val):
        sim1_resnik = s1.res_similarity(s1, self.brown_ic)
        sim2_resnik = s2.res_similarity(s2, self.brown_ic)
        sim_resnik = s1.res_similarity(s2, self.brown_ic) / max(sim1_resnik, sim2_resnik)
        if sim_resnik > val:
            return sim_resnik
        return val

    def jiang(self, s1, s2, val):
        sim1_jiang = s1.jcn_similarity(s1, self.brown_ic)
        sim2_jiang = s2.jcn_similarity(s2, self.brown_ic)
        sim_jiang = s1.jcn_similarity(s2, self.brown_ic) / max(sim1_jiang, sim2_jiang)
        if sim_jiang > val:
            return sim_jiang
        return val
