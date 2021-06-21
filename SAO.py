# -*- coding: UTF-8 -*-
# @CreateTime: 2021/6/15
# @LastUpdateTime: 2021/6/21
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


class SAO:
    def __init__(self, SAOExtracted, WordVector):
        self.pair = None
        SAOFile = FileProcess(SAOExtracted)
        self.SAODict = SAOFile.to_dict()
        vecFile = FileProcess(WordVector)
        self.words_vector, self.vsm_index = vecFile.to_vec()
        self.methods = ['dice', 'inclusion', 'jaccard', 'euclidean', 'pearson', 'spearman', 'arccos', 'Lin', 'resnik',
                        'jiang']

    # id - {S, O, A lists}
    def format_2(self):
        out_dict, patent_label, word_TF, vec_dict = {}, {}, {}, {}  # word_TF: patent_id - word - count
        vec_dict_all = []
        for i in self.pair:
            word_TF[i], label = {}, {}
            out_dict[i], vec_dict[i] = [], []
            text = re.split(';', self.pair[i])
            S, A, O = [], [], []
            for n, j in enumerate(text):

                if j[:3] == '^_^':
                    j = j[3:]
                    label[n] = 1
                else:
                    label[n] = 0
                # split SAO
                SAO = j.split(', ')
                S.append(SAO[0][1:])
                A.append(SAO[1])
                O.append(SAO[2][:-1])
                word = (SAO[0][1:], SAO[1], SAO[2][:-1])

                if SAO[0][1:] == '':
                    vS = [0.0] * 300
                else:
                    temp_wordS = SAO[0][1:].split(' ')
                    vS = self.words_vector[temp_wordS[0]]
                    if len(temp_wordS) > 1:
                        for t1 in temp_wordS[1:]:
                            if t1 != '':
                                vS = [vS[t] + self.words_vector[t1][t] for t in range(300)]
                if word[0] not in vec_dict.keys():
                    vec_dict[word[0]] = vS

                if SAO[1] == '':
                    vA = [0.0] * 300
                else:
                    temp_wordA = SAO[1].split(' ')
                    vA = self.words_vector[temp_wordA[0]]
                    if len(temp_wordA) > 1:
                        for t2 in temp_wordA[1:]:
                            if t2 != "":
                                vA = [vA[t] + self.words_vector[t2][t] for t in range(300)]
                if word[1] not in vec_dict.keys():
                    vec_dict[word[1]] = vA

                if SAO[2][:-1] == '':
                    vO = [0.0] * 300
                else:
                    temp_wordO = SAO[2][:-1].split(' ')
                    vO = self.words_vector[temp_wordO[0]]
                    if len(temp_wordO) > 1:
                        for t2 in temp_wordO[1:]:
                            if t2 != "":
                                vO = [vO[t] + self.words_vector[t2][t] for t in range(300)]
                if word[2] not in vec_dict.keys():
                    vec_dict[word[2]] = vO

                if vS is not None and vO is not None:
                    v = [vS[t] + vA[t] + vO[t] for t in range(300)]
                elif vS is not None:
                    v = [vS[t] + vA[t] for t in range(300)]
                elif vO is not None:
                    v = [vA[t] + vO[t] for t in range(300)]
                else:
                    v = vA
                vec_dict_all.append(v)

                # TF
                if word in word_TF[i]:
                    word_TF[i][word] += 1
                else:
                    word_TF[i][word] = 1
            out_dict[i].append(S)
            out_dict[i].append(A)
            out_dict[i].append(O)
            patent_label[i] = label

        return out_dict, patent_label, word_TF, vec_dict, vec_dict_all

    def new_all(self):
        # weight = [0.2, 0.5, 0.8]
        # thre = [0.2, 0.5, 0.8]

        print("start calculating.....")

        weight_m = ['km', 'sc', 'graph', 'tfidf', 'bm25']
        cleaned_sao, label, TF_count, vec, vec_dict_all = self.format_2()
        weightSys = Weight(self.SAODict, self.pair, vec_dict_all, cleaned_sao)
        weightSys.set_up()
        print(len(weightSys.graph_degrees[1]))

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
        weightSys = Weight(self.SAODict, self.pair, vec_dict_all, cleaned_sao)
        weightSys.set_up()
        for ii, j in enumerate(S[0]):
            # get TFIDF
            w1 = weightSys.tfidf_(j[0], first_SAO, first_ID, TF_count)
            w2 = weightSys.tfidf_(j[1], second_SAO, second_ID, TF_count)
            # get bm25
            matrix, mean = weightSys.bm25_matrix, weightSys.bm25_mean
            # get km weight
            km_weight = 1 if weightSys.km_labels[j[0]] == weightSys.km_labels[len(first_SAO[0]) + j[1]] else 0
            # get sc weight
            sc_weight = 1 if weightSys.sc_labels[j[0]] == weightSys.sc_labels[len(first_SAO[0]) + j[1]] else 0
            # get graph degrees
            d1 = weightSys.graph_degrees[0][j[0]]
            d2 = weightSys.graph_degrees[1][j[1]]
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
            if matrix[j[1]][j[0]] > mean:
                if 'bm25' in score[0].keys():
                    for index in range(len(self.methods)):
                        score[index]['bm25'] += SAO[index][j]
                else:
                    for index in range(len(self.methods)):
                        score[index]['bm25'] = SAO[index][j]
            # km
            if km_weight == 1:
                km_score = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            else:
                km_score = temp_score
            if 'km' in score[0].keys():
                for index in range(len(self.methods)):
                    score[index]['km'] += km_score[index]
            else:
                for index in range(len(self.methods)):
                    score[index]['km'] = km_score[index]

            # sc
            if sc_weight == 1:
                sc_score = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            else:
                sc_score = temp_score
            if 'sc' in score[0].keys():
                for index in range(len(self.methods)):
                    score[index]['sc'] += sc_score[index]
            else:
                for index in range(len(self.methods)):
                    score[index]['sc'] = sc_score[index]

            # graph
            if 'graph' in score[0].keys():
                for index in range(len(self.methods)):
                    score[index]['graph'] += temp_score[
                                                 index] * (d1 + d2) / 2
            else:
                for index in range(len(self.methods)):
                    score[index]['graph'] = temp_score[
                                                index] * (d1 + d2) / 2

            # tfidf
            if 'tfidf' in score[0].keys():
                for index in range(len(self.methods)):
                    score[index]['tfidf'] += temp_score[index] * (w1 * w2)
            else:
                for index in range(len(self.methods)):
                    score[index]['tfidf'] = temp_score[index] * (w1 * w2)
        print("weight time: ", time.time() - t1)
        for index in range(len(self.methods)):
            for m in weight_m:
                score[index][m] = score[index][m] / len(list(S[0].keys()))
        return score

    def main(self):
        # weight = [0.2, 0.5, 0.8]
        weight_m = ['km', 'sc', 'graph', 'tfidf', 'bm25']
        file = {}
        for w in weight_m:
            file[w] = [[], [], [], [], [], [], [], [], [], []]
        compare = []
        target = []
        id_list = []
        for ind, self.pair in enumerate(self.SAODict):
            print('------ #' + str(ind + 1) + ' ----- ', format(ind / len(self.SAODict) * 100, '.2f'), '% done --------')

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
            data.to_csv("ResultFiles/SAO_mean_" + f + ".csv", encoding='utf_8_sig')

