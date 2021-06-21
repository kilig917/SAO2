# -*- coding: UTF-8 -*-
# @CreateTime: 2021/6/21
# @LastUpdateTime: 2021/6/21
# @Author: Yingtong Hu

"""

Top level file for Patent Matching Project

"""

from SAO import SAO


# Get input: extraction_method_1.txt, vector_en_method_1_SAO_glove_array.txt
SAOExtracted = input("Please Enter the Extracted SAO file: ")
WordVector = input("Please Enter the vector file: ")

SAOSys = SAO(SAOExtracted, WordVector)
SAOSys.main()