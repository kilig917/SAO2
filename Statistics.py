# -*- coding: UTF-8 -*-
# @CreateTime: 2021/6/21
# @LastUpdateTime: 2021/6/21
# @Author: Yingtong Hu

"""

Statistic Methods: Mean, Mode, Medium...

"""

import numpy as np


class Statistics:
    def __init__(self, data):
        self.data = data

    def mean(self):
        meanArr = []
        for numArr in self.data:
            meanArr.append(np.mean(numArr))
        return meanArr

    def median(self):
        medianArr = []
        for numArr in self.data:
            medianArr.append(np.median(numArr))
        return medianArr

    def mode(self):
        modeArr = []
        for numArr in self.data:
            tempArr = [float(format(i, '.1f')) for i in numArr]
            d = {}
            for i in tempArr:
                d[i] = tempArr.count(i)
            modeArr.append(max(d, key=d.get))
        return modeArr

    def std(self):
        stdArr = []
        for numArr in self.data:
            stdArr.append(np.std(numArr))
        return stdArr
