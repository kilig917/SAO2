# -*- coding: UTF-8 -*-
# @Time: 2021/6/18
# @Author: Yingtong Hu

"""

Process different types of file

"""


class FileProcess(object):
    def __init__(self, docName):
        self.docName = docName

    def to_dict(self):
        """
        Convert SAOs from txt to dictionary data type
        {patent ID: [SAO list], ...}
        """
        inp = open(self.docName, 'r', encoding='utf-8')
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
        f = open(self.docName, 'r', encoding='utf-8')
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

