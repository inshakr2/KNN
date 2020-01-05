# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 18:09:10 2020

@author: ChangYeol
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# x와 y값을 입력으로 당도, 아삭함을 설정할 수 있다.
# 3인수자리에 k 값을 설정 할 수 있다. default = 3

def knn(n,m,k=3):
    x = pd.read_csv('c:/data/food.csv')
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(x.iloc[:,1:3], x['class'])
    return clf.predict(np.array([[n,m]]))
