# coding=utf-8
# @CreateTime: 2021/6/22
# @LastUpdateTime: 2021/6/23
# @Author: Yingtong Hu

import re
import time
import numpy as np
from pandas.core.frame import DataFrame
from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic
from FileProcess import FileProcess


in_file = "extraction_method_1_100_dic.txt"
in_vector = "vector_en_method_1_SAO_glove_array.txt"
in_grade = "dataset_eng_knowledge_3_8.txt"

print("running MAP_MRR_NDCG.py")


FileSys = FileProcess(vectorFile=in_vector, gradeFile=in_grade, dataFile=in_file)
grade = FileSys.get_grade()
relation, all_ID = FileSys.get_data()
words_vector, vsm_index = FileSys.to_vec()

words_vector[''] = [0.0] * 300
brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')
errorpos = ['a', 'r', 's']

print("after data processing")


def sim_sao(input_patent_SAO):
    cleaned_sao, label, TF_count, vec = format_2(input_patent_SAO)
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
                    # if (label[first_ID][k] == 1 and label[second_ID][ind] == 1) or (
                    #         label[first_ID][k] == 0 and label[second_ID][ind] == 0):
                    v1 = vec[word1]
                    v2 = vec[word2]
                    d_pearson = pearson(v1, v2)
                    d_spearman = spearman_(v1, v2)
                    if np.isnan(d_pearson):
                        d_pearson = 1.0
                    if np.isnan(d_spearman):
                        d_spearman = 1.0
                    d = [dice_coefficient(word1, word2, 0), in_ind(word1, word2), j_cal(word1, word2),
                         euclidean_distance(v1, v2), d_pearson,
                         d_spearman, a_cos(v1, v2)] + hierarchy(word1, word2)
                    # else:
                    #     d = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    if i == 0 and j == 0:
                        for index in range(len(methods)):
                            S[index][(k, ind)] = d[index]
                    if i == 1 and j == 1:
                        for index in range(len(methods)):
                            O[index][(k, ind)] = d[index]
                    if i == 0 and j == 1:
                        for index in range(len(methods)):
                            SO[index][(k, ind)] = d[index]
                    if i == 1 and j == 0:
                        for index in range(len(methods)):
                            OS[index][(k, ind)] = d[index]

    # so os
    A = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
    for k, word1 in enumerate(first_SAO[1]):
        for ind, word2 in enumerate(second_SAO[1]):
            # if (label[first_ID][k] == 1 and label[second_ID][ind] == 1) or (
            #         label[first_ID][k] == 0 and label[second_ID][ind] == 0):
            d = [dice_coefficient(word1, word2, 1)]
            if word2 in word1 or word1 in word2:
                d += [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            else:
                v1 = vec[word1]
                v2 = vec[word2]
                d_pearson = pearson(v1, v2)
                d_spearman = spearman_(v1, v2)
                if np.isnan(d_pearson):
                    d_pearson = 1.0
                if np.isnan(d_spearman):
                    d_spearman = 1.0
                d += [in_ind(word1, word2), j_cal(word1, word2), euclidean_distance(v1, v2), d_pearson,
                      d_spearman, a_cos(v1, v2)] + hierarchy(word1, word2)
            # else:
            #     d = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            for index in range(len(methods)):
                A[index][(k, ind)] = d[index]

    SAO = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
    score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for j in S[0]:
        # get similarity
        if S[0][j] is not False and O[0][j] is not False and SO[0][j] is not False and OS[0][j] is not False:
            for index in range(len(methods)):
                temp1 = S[index][j] + O[index][j]
                temp2 = SO[index][j] + OS[index][j]
                SAO[index][j] = 0.6 * ((max(temp1, temp2)) / 2.0) + 0.4 * A[index][j]
        elif S[0][j] is not False:
            for index in range(len(methods)):
                SAO[index][j] = 0.5 * S[index][j] + 0.5 * A[index][j]
        elif O[0][j] is not False:
            for index in range(len(methods)):
                SAO[index][j] = 0.5 * A[index][j] + 0.5 * O[index][j]
        else:
            for index in range(len(methods)):
                SAO[index][j] = A[index][j]

        for index in range(len(methods)):
            score[index] += SAO[index][j]
    for index in range(len(methods)):
        score[index] = score[index] / len(list(S[0].keys()))
    return score


def MAP_main():
    methods = ['dice', 'inclusion', 'jaccard', 'euclidean', 'pearson', 'spearman', 'arccos', 'Lin', 'resnik', 'jiang']
    record = {}
    mean = [[], [], [], [], [], [], [], [], [], []]
    ncdg = [[], [], [], [], [], [], [], [], [], []]
    MRR_val = [[], [], [], [], [], [], [], [], [], []]
    for index, main_ID in enumerate(relation):
        print('#' + str(index+1) + ', ' + str(len(relation)-index-1) + ' left')
        print("main ID:", main_ID)
        sim_all = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
        for ID in all_ID:
            if ID == main_ID:
                continue
            sim, record = calculate(ID, main_ID, record)
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
            ncdg[i].append(NDCG(rank[i], main_ID))
        # calculate mean
        for i in range(len(methods)):
            if len(targets_rank[i]) != 0:
                sum_ = 0.0
                for ind, val in enumerate(targets_rank[i]):
                    sum_ += (ind + 1) / val
                mean[i].append(sum_ / len(relation[main_ID]))

    # calculate MAP and write
    file = open("MAP_MRR_NDCG_result_3_27_sao.txt", 'a', encoding='utf-8')
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


def calculate(ID, main_ID, record):
    if (main_ID, ID) in record.keys():
        sim = record[(main_ID, ID)]
    elif (ID, main_ID) in record.keys():
        sim = record[(ID, main_ID)]
    else:
        pair = {main_ID: all_ID[main_ID], ID: all_ID[ID]}
        sim = sim_sao(pair)
        record[(main_ID, ID)] = sim
    return sim, record


def NDCG(rank, main_ID):
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


s = time.time()
methods = ['dice', 'inclusion', 'jaccard', 'euclidean', 'pearson', 'spearman', 'arccos', 'Lin', 'resnik', 'jiang']
MAP_main()
e = time.time()
print('time: ', e-s)
