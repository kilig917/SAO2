# -*- coding: UTF-8 -*-
# @CreateTime: 2021/6/15
# @LastUpdateTime: 2021/6/22
# @Author: Yingtong Hu

"""

Calculate SAO using string, concept and vector comparison

"""

import re
import time
from pandas.core.frame import DataFrame
from Statistics import Statistics
from FileProcess import FileProcess
from Formula import Formula
from Weight import Weight


def SAOLabel(phrase):
    if phrase[:3] == '^_^':
        updatePhrase = phrase[3:]
        return 1, updatePhrase
    return 0, phrase


def splitSAO(phrase):
    updatePhrase = phrase.split(', ')
    return updatePhrase[0][1:], updatePhrase[1], updatePhrase[2][:-1]


class SAO:
    def __init__(self, SAOExtracted, WordVector, vectorLen):
        self.pair = None
        SAOFile = FileProcess(SAOExtracted)
        self.SAODict = SAOFile.to_dict()
        vecFile = FileProcess(WordVector)
        self.words_vector, self.vsm_index = vecFile.to_vec()
        self.methods = ['dice', 'inclusion', 'jaccard', 'euclidean', 'pearson', 'spearman', 'arccos', 'Lin', 'resnik',
                        'jiang']
        self.vectorLen = vectorLen

    # id - {S, O, A lists}
    def format_2(self):
        out_dict, patent_label, word_TF, vec_dict = {}, {}, {}, {}  # word_TF: patent_id - word - count
        vec_dict_all = []
        for i in self.pair:
            word_TF[i], label = {}, {}
            vec_dict[i] = []
            text = re.split(';', self.pair[i])
            S, A, O = [], [], []
            for n, j in enumerate(text):

                label[n], j = self.SAOLabel(j)
                s, a, o = self.splitSAO(j)
                S.append(s)
                A.append(a)
                O.append(o)

                if s not in vec_dict.keys():
                    vec_dict[s] = self.__vector(s)

                if a not in vec_dict.keys():
                    vec_dict[a] = self.__vector(a)

                if o not in vec_dict.keys():
                    vec_dict[o] = self.__vector(o)

                v = [vec_dict[s][t] + vec_dict[a][t] + vec_dict[o][t] for t in range(self.vectorLen)]
                vec_dict_all.append(v)

                # TF
                if (s, a, o) in word_TF[i]:
                    word_TF[i][(s, a, o)] += 1
                else:
                    word_TF[i][(s, a, o)] = 1

            out_dict[i] = [S, A, O]
            patent_label[i] = label

        return out_dict, patent_label, word_TF, vec_dict, vec_dict_all

    def __vector(self, word):
        if word != '':
            temp_wordS = word.split(' ')
            v = self.words_vector[temp_wordS[0]]
            if len(temp_wordS) > 1:
                for t1 in temp_wordS[1:]:
                    if t1 != '':
                        v = [v[t] + self.words_vector[t1][t] for t in range(self.vectorLen)]
            return v
        return [0.0] * self.vectorLen

    def new_all(self):
        # weight = [0.2, 0.5, 0.8]
        # thre = [0.2, 0.5, 0.8]

        print("start calculating.....")

        weight_m = ['km', 'sc', 'graph', 'tfidf', 'bm25']
        cleaned_sao, label, TF_count, vec, vec_dict_all = self.format_2()
        weightSys = Weight(self.SAODict, self.pair, vec_dict_all, cleaned_sao, self.methods)
        weightSys.set_up()

        first_ID = next(iter(cleaned_sao))
        first_SAO = cleaned_sao[first_ID]
        cleaned_sao.pop(first_ID)
        second_ID = next(iter(cleaned_sao))
        second_SAO = cleaned_sao[second_ID]
        S = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
        O = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
        SO = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
        OS = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
        first_SO = [first_SAO[0], first_SAO[2]]
        second_SO = [second_SAO[0], second_SAO[2]]
        wordComb = {}

        print("calculating S and O....")
        t1 = time.time()
        for i, l1 in enumerate(first_SO):
            for j, l2 in enumerate(second_SO):
                for k, word1 in enumerate(l1):
                    for ind, word2 in enumerate(l2):
                        if word2 == '' or word1 == '':
                            if i == 0 and j == 0:
                                S[0][(k, ind)] = False
                            if i == 1 and j == 1:
                                O[0][(k, ind)] = False
                            if i == 0 and j == 1:
                                SO[0][(k, ind)] = False
                            if i == 1 and j == 0:
                                OS[0][(k, ind)] = False
                            continue
                        if word1 == word2:
                            d = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                        elif (word1, word2) in wordComb:
                            d = wordComb[(word1, word2)]
                        elif (word2, word1) in wordComb:
                            d = wordComb[(word2, word1)]
                        elif (label[first_ID][k] == 1 and label[second_ID][ind] == 1) or (
                                label[first_ID][k] == 0 and label[second_ID][ind] == 0):
                            v1 = vec[word1]
                            v2 = vec[word2]
                            vectorF = Formula(v1, v2, 'v')
                            wordF = Formula(word1, word2, 'w')
                            hierarchyF = Formula(word1, word2, 'h')
                            d = [wordF.dice(), wordF.inclusionIndex(), wordF.jaccard(), vectorF.euclidean(),
                                 vectorF.pearson(), vectorF.spearman(), vectorF.arcosine(v1, v2)] \
                                + hierarchyF.hierarchy(word1, word2)
                        else:
                            d = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        wordComb[(word1, word2)] = d
                        if i == 0 and j == 0:
                            for index in range(len(self.methods)):
                                S[index][(k, ind)] = d[index]
                        if i == 1 and j == 1:
                            for index in range(len(self.methods)):
                                O[index][(k, ind)] = d[index]
                        if i == 0 and j == 1:
                            for index in range(len(self.methods)):
                                SO[index][(k, ind)] = d[index]
                        if i == 1 and j == 0:
                            for index in range(len(self.methods)):
                                OS[index][(k, ind)] = d[index]
        print("SO time: ", time.time() - t1)
        print("calculating A....")
        t1 = time.time()
        A = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
        wordComb = {}
        for k, word1 in enumerate(first_SAO[1]):
            for ind, word2 in enumerate(second_SAO[1]):
                if word1 == word2:
                    d = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                elif (word1, word2) in wordComb:
                    d = wordComb[(word1, word2)]
                elif (word2, word1) in wordComb:
                    d = wordComb[(word2, word1)]
                elif (label[first_ID][k] == 1 and label[second_ID][ind] == 1) or (
                        label[first_ID][k] == 0 and label[second_ID][ind] == 0):
                    d = []
                    if word2 in word1 or word1 in word2:
                        d += [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                    else:
                        v1 = vec[word1]
                        v2 = vec[word2]
                        vectorF = Formula(v1, v2, 'v')
                        wordF = Formula(word1, word2, 'w')
                        hierarchyF = Formula(word1, word2, 'h')
                        d += [wordF.dice(), wordF.inclusionIndex(), wordF.jaccard(), vectorF.euclidean(),
                              vectorF.pearson(),
                              vectorF.spearman(), vectorF.arcosine(v1, v2)] + hierarchyF.hierarchy(word1, word2)
                else:
                    d = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                wordComb[(word1, word2)] = d
                for index in range(len(self.methods)):
                    A[index][(k, ind)] = d[index]
        print("A time: ", time.time() - t1)
        print("calculating weight....")
        t1 = time.time()
        SAO = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
        score = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
        for ii, j in enumerate(S[0]):
            # get similarity
            if S[0][j] is not False and O[0][j] is not False and SO[0][j] is not False and OS[0][j] is not False:
                for index in range(len(self.methods)):
                    temp1 = S[index][j] + O[index][j]
                    temp2 = SO[index][j] + OS[index][j]
                    SAO[index][j] = 0.6 * ((max(temp1, temp2)) / 2.0) + 0.4 * A[index][j]
            elif S[0][j] is not False:
                for index in range(len(self.methods)):
                    SAO[index][j] = 0.5 * S[index][j] + 0.5 * A[index][j]
            elif O[0][j] is not False:
                for index in range(len(self.methods)):
                    SAO[index][j] = 0.5 * A[index][j] + 0.5 * O[index][j]
            else:
                for index in range(len(self.methods)):
                    SAO[index][j] = A[index][j]

            temp_score = []
            for index in range(len(self.methods)):
                temp_score.append(SAO[index][j])

            # bm25
            score = weightSys.bm25(j, SAO, score)

            # km
            score = weightSys.KMeans(j, first_SAO, temp_score, score)

            # sc
            score = weightSys.SpectralClustering(j, first_SAO, temp_score, score)

            # graph
            score = weightSys.Graph(j, temp_score, score)

            # tfidf
            score = weightSys.tfidf(j, first_SAO, second_SAO, first_ID, second_ID, TF_count, temp_score, score)

        print("weight time: ", time.time() - t1)
        for index in range(len(self.methods)):
            for m in weight_m:
                score[index][m] = score[index][m] / len(list(S[0].keys()))
        return score

    def main(self):
        weight_m = ['km', 'sc', 'graph', 'tfidf', 'bm25']
        file = {}
        for w in weight_m:
            file[w] = [[], [], [], [], [], [], [], [], [], []]
        compare, target, id_list = [], [], []
        for ind, self.pair in enumerate(self.SAODict):
            print('------ #' + str(ind + 1) + ' ----- ', format(ind / len(self.SAODict) * 100, '.2f'),
                  '% done --------')

            id_ = list(self.pair.keys())
            if id_ in id_list or len(id_) != 2:
                print('id error!\n')
                continue
            id_list.append(id_)
            compare.append(id_[0])
            target.append(id_[1])

            score = self.new_all()

            for i in range(len(score)):
                for key in score[i].keys():
                    file[key][i].append(score[i][key])

            break

        compare += ["", "mean", "median", "mode", "standard deviation"]
        for f in file:
            StatSys = Statistics(file[f])
            mean, median, mode, std = StatSys.mean(), StatSys.median(), StatSys.mode(), StatSys.std()

            for i in range(len(file[f])):
                file[f][i] += ["", mean[i], median[i], mode[i], std[i]]
            data = [compare, target + [""]] + file[f]

            data = DataFrame(data)
            data = data.T
            data.rename(
                columns={0: 'compare_ID', 1: 'target_ID', 2: 'Dice', 3: 'Inclusion', 4: 'Jac', 5: 'Euclidean',
                         6: 'Pearson',
                         7: 'Spearman', 8: 'Arccos', 9: 'Lin', 10: 'Resnik',
                         11: 'Jiang'
                         }, inplace=True)
            data.to_csv("ResultFiles/test" + f + ".csv", encoding='utf_8_sig')
