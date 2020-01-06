# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 18:08:58 2020

@author: ChangYeol
"""

import numpy as np
import operator

# 기준점을 지정
point = [2,1]
# 데이터를 임의로 할당
pointlist = [(1,1),(1,0),(2,0),(0,1),(2,2),(1,5),(2,3)]


def knn(point,pointlist,k):
    dic = {}
    for p in pointlist:
        distance = np.sqrt(np.sum(pow(np.array(point) - np.array(p),2)))
        dic[p] = distance
    
    sort_dic = sorted(dic.items(), key = operator.itemgetter(1))
    res = []
    
    for key in sort_dic:
        if len(res) < k:
            res.append(key[0])
    
    return res
