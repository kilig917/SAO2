# -*- coding: UTF-8 -*-
# @CreateTime: 2021/6/21
# @LastUpdateTime: 2021/6/24
# @Author: Yingtong Hu

"""

Top level file for Patent Matching Project

"""

from SAO import SAO


# Get input: extraction_method_1.txt, vector_en_method_1_SAO_glove_array.txt
# testSAO.txt
# SAOExtracted = input("Please Enter the Extracted SAO file: ")
# WordVector = input("Please Enter the vector file: ")

SAOExtracted = 'testSAO.txt'
WordVector = 'vector_en_method_1_SAO_glove_array.txt'
vectorLen = 300

SAOSys = SAO(SAOExtracted, WordVector, vectorLen, 1)
SAOSys.main()
SAOSys.MAP_main()
