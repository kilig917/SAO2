# -*- coding: UTF-8 -*-
# @CreateTime: 2021/6/18
# @LastUpdateTime: 2021/6/22
# @Author: Yingtong Hu

"""

Process different types of file

"""


class FileProcess:
    def __init__(self, vectorFile, extractFile="", gradeFile="", dataFile=""):
        self.extractFile = extractFile
        self.vectorFile = vectorFile
        self.gradeFile = gradeFile
        self.dataFile = dataFile

    def to_dict(self):
        """
        Convert SAOs from txt to dictionary data type
        [{patent ID1: SAO string, patent ID2: SAO string}, {}...]
        """
        inp = open(self.extractFile, 'r', encoding='utf-8')
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

    def to_vec(self):
        """
        Convert vectors from txt to array data type
        [[vec1], [vec2]...]
        """
        f = open(self.vectorFile, 'r', encoding='utf-8')
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

    def get_grade(self):
        file = open(self.gradeFile, 'r', encoding='utf-8')
        line = file.readline()
        dic = {}
        while line:
            info = line.strip().split(' ')
            ID_1 = info[1]
            ID_2 = info[2]
            grade = 2 ** int(info[3]) - 1
            if (ID_1, ID_2) not in dic.keys() and (ID_2, ID_1) not in dic.keys():
                dic[(ID_1, ID_2)] = grade
            line = file.readline()
        return dic

    def get_data(self):
        file = open(self.dataFile, 'r', encoding='utf-8')
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




