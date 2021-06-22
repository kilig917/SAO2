# coding=utf-8
# @CreateTime: 2021/6/22
# @LastUpdateTime: 2021/6/22
# @Author: Yingtong Hu

import re
import time
import numpy as np
from pandas.core.frame import DataFrame
# from fasttext import FastText
from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic


in_file_recall = "extraction_method_1.txt"
# in_file = "ipc_100_dic.txt"
# in_file = "field_tech_dic_100.txt"
in_file = "extraction_method_1_100_dic.txt"
in_vector = "vector_en_method_1_SAO_glove_array.txt"
in_grade = "dataset_eng_knowledge_3_8.txt"

print("running MAP_MRR_NDCG.py")


def file_to_dict_2(in_file_name, dic):
    inp = open(in_file_name, 'r', encoding='utf-8')
    line = inp.readline()
    d = {}
    while line:
        if line.strip() == "":
            line = inp.readline()
            continue
        key, info = line.split(':', 1)
        if d.__contains__(key):
            d[key] += ";" + info.strip()
        else:
            d[key] = info.strip()
        line = inp.readline()
    out_list = []
    first_ID = next(iter(dic))
    first_SAO = dic[first_ID]
    for key in d.keys():
        out_list.append({first_ID: first_SAO, key: d[key]})
    return out_list


def file_to_dict(in_file_name):
    inp = open(in_file_name, 'r', encoding='utf-8')
    text_line = inp.readline()
    out_list = []
    temp_dic = {}
    while text_line:
        if text_line.strip() == "":
            text_line = inp.readline()
            continue
        if "===========" in text_line:
            out_list.append(temp_dic)
            temp_dic = {}
            text_line = inp.readline()
            continue
        key, info = text_line.split(":", 1)
        if temp_dic.__contains__(key):
            temp_dic[key] += ";" + info.strip()
        else:
            temp_dic[key] = info.strip()
        text_line = inp.readline()
    return out_list


def file_to_vec(file):
    f = open(file, 'r', encoding='utf-8')
    line = f.readline()
    vector = {}
    vsm_ind = {}
    count = 0
    while line:
        line = line.strip().split(':')
        if line[0] not in vector.keys():
            vector[line[0]] = [float(i) for i in line[1].split(" ")]
            vsm_ind[line[0]] = count
            count += 1
        line = f.readline()
    return vector, vsm_ind


def IDF(input_patent_SAO):
    word_IDF = {}  # word - [patent]
    patent_ID = []
    for i in input_patent_SAO:
        text = re.split(';', input_patent_SAO[i])
        for n, j in enumerate(text):
            if j[:3] == '^_^':
                j = j[3:]
            # split SAO
            SAO = j.split('#')
            word = (SAO[0][1:], SAO[1], SAO[2][:-1])
            # IDF
            if word in word_IDF:
                if i not in word_IDF[word]:
                    word_IDF[word].append(i)
            else:
                word_IDF[word] = [i]
    return word_IDF, len(input_patent_SAO)


def get_data(file):
    r = {}
    allID = {}
    line = file.readline()
    while line:
        line = line.strip().split('$')
        ID_1, info_1 = line[0].split(': ', 1)
        ID_2, info_2 = line[1].split(': ', 1)
        if ID_2 in r.keys():
            r[ID_2].append(ID_1)
        else:
            r[ID_2] = [ID_1]

        if ID_1 not in allID.keys():
            allID[ID_1] = info_1
        if ID_2 not in allID.keys():
            allID[ID_2] = info_2

        line = file.readline()
    return r, allID

def get_grade(f):
    file = open(f, 'r', encoding='utf-8')
    line = file.readline()
    dic = {}
    while line:
        info = line.strip().split(' ')
        ID_1 = info[1]
        ID_2 = info[2]
        grade = 2**int(info[3]) - 1
        if (ID_1, ID_2) not in dic.keys() and (ID_2, ID_1) not in dic.keys():
            dic[(ID_1, ID_2)] = grade
        line = file.readline()
    return dic


grade = get_grade(in_grade)
f = open(in_file, 'r', encoding='utf-8')
relation, all_ID = get_data(f)
# print(all_ID)

words_vector, vsm_index = file_to_vec(in_vector)
words_vector[''] = [0.0] * 300
# IDF_count, doc_num = IDF_ft(all_ID)
brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')
errorpos = ['a', 'r', 's']

print("after data processing")

def dice_similarity_ipc(input_patent_SAO):
    IDs = list(input_patent_SAO.keys())
    first_ID = IDs[0]
    first_ipc = input_patent_SAO[first_ID]
    second_ID = IDs[1]
    second_ipc = input_patent_SAO[second_ID]
    score = dice_coefficient_ipc(first_ipc, second_ipc, 0)

    sim = []
    total = 1
    sim.append(second_ID + '(' + str(score / total) + ', ' + str(score) + ')')
    return sim


# calculate dice coefficient between two strings
def dice_coefficient_ipc(a, b, is_A):
    """dice coefficient 2nt/na + nb."""
    if is_A:
        if a in b or b in a:
            return 1
    a_ = set(a)
    b_= set(b)
    overlap = len(a_ & b_)
    d = overlap * 2.0 / (len(a_) + len(b_))
    return d


def inclusion_index_ipc(input_patent_SAO):
    IDs = list(input_patent_SAO.keys())
    first_ID = IDs[0]
    first_ipc = input_patent_SAO[first_ID]
    second_ID = IDs[1]
    second_ipc = input_patent_SAO[second_ID]
    score = in_ind_ipc(first_ipc, second_ipc)

    sim = []
    total = 1
    sim.append(second_ID + '(' + str(score / total) + ', ' + str(score) + ')')
    return sim


def in_ind_ipc(w1, w2):
    a = set(w1)
    b = set(w2)
    common = a & b
    s = len(common) / min(len(a), len(b))
    return s


def j_cal_ipc(w1, w2):
    a = set(w1)
    b = set(w2)
    s = len(set.intersection(a, b)) / len(set.union(a, b))
    return s


def ja_ipc(input_patent_SAO):
    IDs = list(input_patent_SAO.keys())
    first_ID = IDs[0]
    first_ipc = input_patent_SAO[first_ID]
    second_ID = IDs[1]
    second_ipc = input_patent_SAO[second_ID]
    score = j_cal_ipc(first_ipc, second_ipc)

    sim = []
    total = 1
    sim.append(second_ID + '(' + str(score / total) + ', ' + str(score) + ')')
    return sim

def sim_ft(input_patent_SAO):
    cleaned_sao = format_2_ft(input_patent_SAO)
    first_ID = next(iter(cleaned_sao))
    first_words = cleaned_sao[first_ID]
    cleaned_sao.pop(first_ID)
    second_ID = next(iter(cleaned_sao))
    second_words = cleaned_sao[second_ID]
    s = [0, 0, 0]
    tt_dict = [{}, {}, {}]
    for word1 in first_words:
        for word2 in second_words:
            if word1 == '' or word2 == '':
                k = [0.0, 0.0, 0.0]
            else:
                k = [dice_coefficient_ft(word1, word2, 0), in_ind_ft(word1, word2), j_cal_ft(word1, word2)]
            for i in range(len(methods)):
                tt_dict[i][(word1, word2)] = k[i]
    # term sorting
    E1, E2 = [[], [], []], [[], [], []]
    ESD = [0.0, 0.0, 0.0]
    for i in range(len(methods)):
        E1[i], E2[i] = term_sorting(first_words, second_words, tt_dict[i])
        # entity-to-entity(S-S, O-S, A-A, S-O, O-O) similarity using extended_sd
        for j in range(min(len(E1[i]), len(E2[i]))):
            ESD[i] += tt_dict[i][(E1[i][j], E2[i][j])]
        ESD[i] = 2 * ESD[i] / (len(E1[i]) + len(E2[i]))
        s[i] = ESD[i]
    return s

# Greedy algorithm for term sorting
# input: E_1 - entity 1
#        E_2 - entity 2
# output: entity 1 and entity 2 with updated term order
def term_sorting(E_1, E_2, d):
    flag_i = 0
    flag_j = 0
    m = len(E_1)
    n = len(E_2)
    for k in range(min(m, n)):
        max_temp = -1
        # search for the k-th maximum matching
        for i in range(k, m):
            for j in range(k, n):
                sim_temp = d[(E_1[i], E_2[j])]  # similarity between term i and term j
                if sim_temp > max_temp:
                    flag_i = i
                    flag_j = j
                    max_temp = sim_temp
        # reorder the terms in E1 and E2
        # swap Term k with Term flag_i for E1
        temp = E_1[k]
        E_1[k] = E_1[flag_i]
        E_1[flag_i] = temp
        # swap Term k with Term flag_j for E2
        temp = E_2[k]
        E_2[k] = E_2[flag_j]
        E_2[flag_j] = temp
    return E_1, E_2

# calculate dice coefficient between two strings
def dice_coefficient_ft(a, b, is_A):
    """dice coefficient 2nt/na + nb."""
    if is_A:
        if a in b or b in a:
            return 1
    a_ = set()
    for ele in a:
        a_.add(ele)
    b_= set()
    for ele in b:
        b_.add(ele)
    overlap = len(a_ & b_)
    if len(a_) + len(b_) == 0:
        return 1
    d = overlap * 2.0 / (len(a_) + len(b_))
    return d


def in_ind_ft(w1, w2):
    a = set()
    for ele in w1:
        a.add(ele)
    b = set()
    for ele in w2:
        b.add(ele)
    if len(a) == 0 or len(b) == 0:
        return 0
    common = a & b
    s = len(common) / min(len(a), len(b))
    return s


def j_cal_ft(w1, w2):
    word1 = w1.split(' ')
    word2 = w2.split(' ')
    a = set()
    for ele in word1:
        a.add(ele)
    b = set()
    for ele in word2:
        b.add(ele)
    if len(set.union(a, b)) == 0:
        return 0
    s = len(set.intersection(a, b)) / len(set.union(a, b))
    return s

def format_2_ft(patent_pair):
    info = {}
    for id_ in patent_pair:
        text = re.split('#', patent_pair[id_])
        text = text[2] + text[4]
        words = text.split(',')
        words.remove('')
        info[id_] = words
    return info


# id - {S, O, A lists}
def format_2(input_patent_SAO):
    out_dict = {}
    patent_label = {}
    word_TF = {}  # patent_id - word - count
    vec_dict_all = []
    vec_dict = {}
    for i in input_patent_SAO:
        word_TF[i] = {}
        out_dict[i] = []
        vec_dict[i] = []
        label = {}
        text = re.split(';', input_patent_SAO[i])
        S = []
        A = []
        O = []
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
                vS = None
            else:
                temp_wordS = SAO[0][1:].split(' ')
                vS = words_vector[temp_wordS[0]]
                if len(temp_wordS) > 1:
                    for t1 in temp_wordS[1:]:
                        if t1 != '':
                            vS = [vS[t] + words_vector[t1][t] for t in range(300)]
            if word[0] not in vec_dict.keys():
                vec_dict[word[0]] = vS

            if SAO[1] == '':
                vA = [0.0] * 300
            else:
                temp_wordA = SAO[1].split(' ')
                vA = words_vector[temp_wordA[0]]
                if len(temp_wordA) > 1:
                    for t2 in temp_wordA[1:]:
                        if t2 != "":
                            vA = [vA[t] + words_vector[t2][t] for t in range(300)]
            if word[1] not in vec_dict.keys():
                vec_dict[word[1]] = vA

            if SAO[2][:-1] == '':
                vO = None
            else:
                temp_wordO = SAO[2][:-1].split(' ')
                vO = words_vector[temp_wordO[0]]
                if len(temp_wordO) > 1:
                    for t2 in temp_wordO[1:]:
                        if t2 != "":
                            vO = [vO[t] + words_vector[t2][t] for t in range(300)]
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

    return out_dict, patent_label, word_TF, vec_dict

def dice_coefficient(a, b, is_A):
    """dice coefficient 2nt/na + nb."""
    if is_A:
        if a in b or b in a:
            return 1
    a_ = set()
    for ele in a:
        a_.add(ele)
    b_ = set()
    for ele in b:
        b_.add(ele)
    overlap = len(a_ & b_)
    d = overlap * 2.0 / (len(a_) + len(b_))
    return d


def in_ind(w1, w2):
    a = set()
    for ele in w1:
        a.add(ele)
    b = set()
    for ele in w2:
        b.add(ele)
    common = a & b
    s = len(common) / min(len(a), len(b))
    return s


def j_cal(w1, w2):
    word1 = w1.split(' ')
    word2 = w2.split(' ')
    a = set()
    for ele in word1:
        a.add(ele)
    b = set()
    for ele in word2:
        b.add(ele)
    s = len(set.intersection(a, b)) / len(set.union(a, b))
    return s


def euclidean_distance(v1, v2):
    V1 = np.array(v1)
    V2 = np.array(v2)
    d = np.sqrt(np.sum(np.square(V1 - V2)))
    return 1 / (1 + d)


def pearson(x, y):
    x = np.array(x)
    y = np.array(y)
    data = DataFrame({'x': x, 'y': y})
    return abs(data.corr()['x']['y'])


def spearman_(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    data = DataFrame({'x': v1, 'y': v2})
    return abs(data.corr(method='spearman')['x']['y'])


def a_cos(v1, v2):
    a = np.sqrt(sum([np.square(x) for x in v1]))
    b = np.sqrt(sum([np.square(x) for x in v2]))
    if a == 0.0 or b == 0.0:
        return 0.0
    d = np.dot(v1, v2) / (a * b)
    return 0.5 * d + 0.5


def hierarchy(word1, word2):
    temp_word1 = word1.split(' ')
    temp_word2 = word2.split(' ')
    d = [0.0, 0.0, 0.0]
    for t1 in temp_word1:
        for t2 in temp_word2:
            result = hierarchy_helper(t1, t2)
            for i in range(3):
                d[i] += result[i]
    return d


def hierarchy_helper(w1, w2):
    if w1 == w2:
        return [1.0, 1.0, 1.0]
    synsets1 = wordnet.synsets(w1)
    synsets2 = wordnet.synsets(w2)
    if not synsets1 or not synsets2:
        return [0.0, 0.0, 0.0]

    maxx = [0.0, 0.0, 0.0]
    for s1 in synsets1:
        for s2 in synsets2:
            # sim_wu = s1.wup_similarity(s2)
            # if sim_wu is not None and sim_wu > maxx[2]:
            #     maxx[2] = sim_wu
            if s1._pos == s2._pos and s1._pos not in errorpos and s2._pos not in errorpos:
                sim_lin = s1.lin_similarity(s2, semcor_ic)
                if sim_lin is not None and sim_lin > maxx[0]:
                    maxx[0] = sim_lin

                # sim1_leacock = s1.lch_similarity(s1)
                # sim2_leacock = s2.lch_similarity(s2)
                # sim_leacock = s1.lch_similarity(s2) / max(sim1_leacock, sim2_leacock)
                # if sim_leacock > maxx[1]:
                #     maxx[1] = sim_leacock

                sim1_resnik = s1.res_similarity(s1, brown_ic)
                sim2_resnik = s2.res_similarity(s2, brown_ic)
                sim_resnik = s1.res_similarity(s2, brown_ic) / max(sim1_resnik, sim2_resnik)
                if sim_resnik > maxx[1]:
                    maxx[1] = sim_resnik

                sim1_jiang = s1.jcn_similarity(s1, brown_ic)
                sim2_jiang = s2.jcn_similarity(s2, brown_ic)
                sim_jiang = s1.jcn_similarity(s2, brown_ic) / max(sim1_jiang, sim2_jiang)
                if sim_jiang > maxx[2]:
                    maxx[2] = sim_jiang

    return maxx

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
        # if label[second_ID][j[1]] == 1:
        #     score[index] += weight[0] * SAO[index][j]
        #     # score_12 += SAO[j] * (w1 * w2)
        # else:
        #     if SAO[j] > thre:
        #         s = 1.0
        #     else:
        #         s = SAO[j]
        #     score_12 += weight[1] * s * (w1 * w2)
        #     # score_12 += SAO[j] * (w1 * w2)
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
