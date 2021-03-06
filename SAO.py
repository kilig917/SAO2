# -*- coding: UTF-8 -*-
# @CreateTime: 2021/6/15
# @LastUpdateTime: 2021/6/24
# @Author: Yingtong Hu

"""

Calculate SAO using string, concept and vector comparison

"""

import re
import time
import numpy as np
from pandas.core.frame import DataFrame
from Statistics import Statistics
from FileProcess import FileProcess
from Formula import Formula
from Weight import Weight

sourceID = ""
targetID = ""
weight_m = ['km', 'sc', 'graph', 'tfidf', 'bm25']


def SAOLabel(phrase):
    if phrase[:3] == '^_^':
        updatePhrase = phrase[3:]
        return 1, updatePhrase
    return 0, phrase


def splitSAO(phrase):
    updatePhrase = phrase.split(', ')
    return updatePhrase[0][1:], updatePhrase[1], updatePhrase[2][:-1]


class SAO:
    def __init__(self, SAOExtracted, WordVector, vectorLen, ifWeight):
        FileSys = FileProcess(extractFile=SAOExtracted, vectorFile=WordVector)
        self.words_vector, self.vsm_index = FileSys.to_vec()
        self.SAODict = FileSys.to_dict()
        self.ifWeight = ifWeight
        self.methods = ['dice', 'inclusion', 'jaccard', 'euclidean', 'pearson', 'spearman', 'arccos', 'Lin', 'resnik',
                        'jiang']
        self.vectorLen = vectorLen
        self.wordComb, self.label, self.phraseVector, self.simSnO = {}, {}, {}, {}
        self.weightScore = {}
        self.score = []

    # id - {S, O, A lists}
    def format_2(self, pair):
        out_dict, self.label, word_TF, self.phraseVector = {}, {}, {}, {}  # word_TF: patent_id - word - count
        vec_dict_all = []
        for i in pair:
            word_TF[i], label = {}, {}
            self.phraseVector[i] = []
            text = re.split(';', pair[i])
            S, A, O = [], [], []
            for n, j in enumerate(text):

                label[n], j = SAOLabel(j)
                s, a, o = splitSAO(j)
                S.append(s)
                A.append(a)
                O.append(o)

                if s not in self.phraseVector.keys():
                    self.phraseVector[s] = self.__vector(s)

                if a not in self.phraseVector.keys():
                    self.phraseVector[a] = self.__vector(a)

                if o not in self.phraseVector.keys():
                    self.phraseVector[o] = self.__vector(o)

                v = [self.phraseVector[s][t] + self.phraseVector[a][t] + self.phraseVector[o][t] for t in
                     range(self.vectorLen)]
                vec_dict_all.append(v)

                # TF
                if (s, a, o) in word_TF[i]:
                    word_TF[i][(s, a, o)] += 1
                else:
                    word_TF[i][(s, a, o)] = 1

            out_dict[i] = [S, A, O]
            self.label[i] = label

        return out_dict, word_TF, vec_dict_all

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

    def new_all(self, pair):
        print("start calculating.....")
        global sourceID, targetID

        cleaned_sao, TF_count, vec_dict_all = self.format_2(pair)

        sourceID = list(cleaned_sao.keys())[0]
        first_SAO = cleaned_sao[sourceID]
        targetID = list(cleaned_sao.keys())[1]
        second_SAO = cleaned_sao[targetID]

        # S: 0, 0   O: 1, 1     SO: 0, 1    OS: 1, 0
        self.simSnO = {(0, 0): [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}],
                       (1, 1): [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}],
                       (1, 0): [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}],
                       (0, 1): [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]}
        first_SO = [first_SAO[0], first_SAO[2]]
        second_SO = [second_SAO[0], second_SAO[2]]

        print("calculating S and O....")
        t1 = time.time()
        for i, l1 in enumerate(first_SO):
            for j, l2 in enumerate(second_SO):
                for ind1, word1 in enumerate(l1):
                    for ind2, word2 in enumerate(l2):
                        if word2 == '' or word1 == '':
                            self.simSnO[(i, j)][0][(ind1, ind2)] = False
                            continue
                        d = self.similarity_SAO(word1, word2, ind1=ind1, ind2=ind2)
                        for index in range(len(self.methods)):
                            self.simSnO[(i, j)][index][(ind1, ind2)] = d[index]
        print("SO time: ", time.time() - t1)
        print("calculating A....")
        t1 = time.time()
        A = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
        for ind1, word1 in enumerate(first_SAO[1]):
            for ind2, word2 in enumerate(second_SAO[1]):
                d = self.similarity_SAO(word1, word2, ind1, ind2, wordType="a")
                for index in range(len(self.methods)):
                    A[index][(ind1, ind2)] = d[index]
        print("A time: ", time.time() - t1)
        print("calculating weight....")
        t1 = time.time()

        SAOSim = self.get_SAO_sim(A)

        if self.ifWeight:
            self.get_sim_w_weight(SAOSim, vec_dict_all, cleaned_sao, first_SAO, second_SAO, TF_count, pair)

        return self.get_patent_sim(SAOSim)

    def similarity_SAO(self, word1, word2, ind1, ind2, wordType=""):
        d = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if self.label[sourceID][ind1] == self.label[targetID][ind2]:
            if (word1 == word2) or (wordType == "a" and (word1 in word2 or word2 in word1)):
                d = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            elif (word1, word2) in self.wordComb:
                d = self.wordComb[(word1, word2)]
            elif (word2, word1) in self.wordComb:
                d = self.wordComb[(word2, word1)]
            else:
                d = self.similarity_2words(word1, word2)
        self.wordComb[(word1, word2)] = d
        return d

    def similarity_2words(self, word1, word2):
        v1, v2 = self.phraseVector[word1], self.phraseVector[word2]
        vectorF, wordF, conceptF = Formula(v1, v2, 'v'), Formula(word1, word2, 'w'), Formula(word1, word2, 'h')
        return [wordF.dice(), wordF.inclusionIndex(), wordF.jaccard(), vectorF.euclidean(),
                vectorF.pearson(), vectorF.spearman(), vectorF.arcosine(v1, v2)] + conceptF.hierarchy(word1, word2)

    def get_SAO_sim(self, A):
        SAOSim = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
        self.score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for j in self.simSnO[(0, 0)][0]:
            for index in range(len(self.methods)):
                if self.simSnO[(0, 0)][0][j] is not False and self.simSnO[(1, 1)][0][j] is not False \
                        and self.simSnO[(0, 1)][0][j] is not False and self.simSnO[(1, 0)][0][j] is not False:
                    temp1 = self.simSnO[(0, 0)][index][j] + self.simSnO[(1, 1)][index][j]
                    temp2 = self.simSnO[(0, 1)][index][j] + self.simSnO[(1, 0)][index][j]
                    SAOSim[index][j] = 0.6 * ((max(temp1, temp2)) / 2.0) + 0.4 * A[index][j]
                elif self.simSnO[(0, 0)][0][j] is not False:
                    SAOSim[index][j] = 0.5 * self.simSnO[(0, 0)][index][j] + 0.5 * A[index][j]
                elif self.simSnO[(1, 1)][0][j] is not False:
                    SAOSim[index][j] = 0.5 * A[index][j] + 0.5 * self.simSnO[(1, 1)][index][j]
                else:
                    SAOSim[index][j] = A[index][j]
            for index in range(len(self.methods)):
                self.score[index] += SAOSim[index][j]
        return SAOSim

    def get_patent_sim(self, SAOSim):
        score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for j in self.simSnO[(0, 0)][0]:
            for index in range(len(self.methods)):
                score[index] += SAOSim[index][j]
        for index in range(len(self.methods)):
            score[index] = score[index] / len(list(self.simSnO[(0, 0)][0].keys()))
        return score

    def get_sim_w_weight(self, SAOSim, vec_dict_all, cleaned_sao, first_SAO, second_SAO, TF_count, pair):
        weightSys = Weight(self.SAODict, pair, vec_dict_all, cleaned_sao, self.methods)
        weightSys.set_up()
        self.weightScore = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
        for j in self.simSnO[(0, 0)][0]:
            temp_score = []
            for index in range(len(self.methods)):
                temp_score.append(SAOSim[index][j])

            # bm25
            self.weightScore = weightSys.bm25(j, SAOSim, self.weightScore)

            # km
            self.weightScore = weightSys.KMeans(j, first_SAO, temp_score, self.weightScore)

            # sc
            self.weightScore = weightSys.SpectralClustering(j, first_SAO, temp_score, self.weightScore)

            # graph
            self.weightScore = weightSys.Graph(j, temp_score, self.weightScore)

            # tfidf
            self.weightScore = weightSys.tfidf(j, first_SAO, second_SAO, TF_count, temp_score, self.weightScore)

        for index in range(len(self.methods)):
            for m in weight_m:
                self.weightScore[index][m] = self.weightScore[index][m] / len(list(self.simSnO[(0, 0)][0].keys()))

    def main(self):
        file = {}
        for w in weight_m:
            file[w] = [[], [], [], [], [], [], [], [], [], []]
        compare, target, id_list = [], [], []
        for ind, pair in enumerate(self.SAODict):
            print('------ #' + str(ind + 1) + ' ----- ', format(ind / len(self.SAODict) * 100, '.2f'),
                  '% done --------')

            id_ = list(pair.keys())
            if id_ in id_list or len(id_) != 2:
                print('id error!\n')
                continue
            id_list.append(id_)
            compare.append(id_[0])
            target.append(id_[1])

            self.new_all(pair)

            for i in range(len(self.weightScore)):
                for key in self.weightScore[i].keys():
                    file[key][i].append(self.weightScore[i][key])

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

    def MAP_main(self):
        in_file = "extraction_method_1_100_dic.txt"
        in_vector = "vector_en_method_1_SAO_glove_array.txt"
        in_grade = "dataset_eng_knowledge_3_8.txt"

        FileSys = FileProcess(vectorFile=in_vector, gradeFile=in_grade, dataFile=in_file)
        grade = FileSys.get_grade()
        relation, all_ID = FileSys.get_data()
        words_vector, vsm_index = FileSys.to_vec()
        words_vector[''] = [0.0] * 300

        methods = ['dice', 'inclusion', 'jaccard', 'euclidean', 'pearson', 'spearman', 'arccos', 'Lin', 'resnik',
                   'jiang']
        record = {}
        mean = [[], [], [], [], [], [], [], [], [], []]
        ncdg = [[], [], [], [], [], [], [], [], [], []]
        MRR_val = [[], [], [], [], [], [], [], [], [], []]
        for index, main_ID in enumerate(relation):
            print('#' + str(index + 1) + ', ' + str(len(relation) - index - 1) + ' left')
            print("main ID:", main_ID)
            sim_all = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
            for ID in all_ID:
                if ID == main_ID:
                    continue
                sim, record = self.calculate(all_ID, ID, main_ID, record)
                if sim is not None:
                    for i in range(len(methods)):
                        sim_all[i][ID] = sim[i]

            # reorder
            rank = [[], [], [], [], [], [], [], [], [], []]
            for i in range(len(methods)):
                while sim_all[i]:
                    closest_patent = max(sim_all[i], key=sim_all[i].get)
                    rank[i].append(closest_patent)
                    sim_all[i].pop(closest_patent)
            # MAP, MRR
            targets_rank = [[], [], [], [], [], [], [], [], [], []]
            for i in range(len(methods)):
                for ID in relation[main_ID]:
                    targets_rank[i].append(rank[i].index(ID) + 1)
                    MRR_val[i].append(1 / (rank[i].index(ID) + 1))
                targets_rank[i].sort()
            # NCDG
            for i in range(len(methods)):
                ncdg[i].append(self.NDCG(grade, rank[i], main_ID))
            # calculate mean
            for i in range(len(methods)):
                if len(targets_rank[i]) != 0:
                    sum_ = 0.0
                    for ind, val in enumerate(targets_rank[i]):
                        sum_ += (ind + 1) / val
                    mean[i].append(sum_ / len(relation[main_ID]))

        # calculate MAP and write
        file = open("MAP_MRR_NDCG_result_6_22_sao.txt", 'a', encoding='utf-8')
        file.write("--------------MAP result------------\n")
        for i in range(len(methods)):
            file.write(methods[i] + ': ')
            m = np.sum(mean[i]) / len(mean[i])
            file.write(str(m) + '\n')
        file.write("\n--------------MRR result------------\n")
        for i in range(len(methods)):
            file.write(methods[i] + ': ')
            m = np.sum(MRR_val[i]) / len(MRR_val[i])
            file.write(str(m) + '\n')
        file.write("\n--------------NDCG result------------\n")
        for i in range(len(methods)):
            file.write(methods[i] + ': ')
            m = np.sum(ncdg[i]) / len(ncdg[i])
            file.write(str(m) + '\n')

    def calculate(self, all_ID, ID, main_ID, record):
        if (main_ID, ID) in record.keys():
            sim = record[(main_ID, ID)]
        elif (ID, main_ID) in record.keys():
            sim = record[(ID, main_ID)]
        else:
            pair = {main_ID: all_ID[main_ID], ID: all_ID[ID]}
            sim = self.new_all(pair)
            record[(main_ID, ID)] = sim
        return sim, record

    def NDCG(self, grade, rank, main_ID):
        grades = []
        DCG = []
        last_dcg = 0.0
        for i, ID in enumerate(rank):
            if (main_ID, ID) in grade.keys():
                gain = grade[(main_ID, ID)]
            else:
                gain = grade[(ID, main_ID)]
            grades.append(gain)
            dcg = last_dcg + gain * np.log(2) / np.log(1 + (i + 1))
            DCG.append(dcg)
            last_dcg = dcg
        grades.sort(reverse=True)
        maxDCG = []
        last_dcg = 0.0
        for i, g in enumerate(grades):
            dcg = last_dcg + g * np.log(2) / np.log(1 + (i + 1))
            maxDCG.append(dcg)
            last_dcg = dcg
        NDCG = 0.0
        for i in range(len(grades)):
            if maxDCG[i] == 0.0:
                NDCG += 0.0
            else:
                NDCG += DCG[i] / maxDCG[i]
        NDCG /= len(grades)
        return NDCG
