# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 17:11:49 2020

@author: ChangYeol
"""


# k값을 입력받아 정확도를 리턴하는 프로그램

def wisc(k):
                    # import section
    import pandas as pd
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    
    
                    # data 추출
    data = pd.read_csv('c:/data/wisc_bc_data.csv')
    data = data.drop('id',axis=1)               # 환자 id는 제거
    data.iloc[:,1:] = scale(data.iloc[:,1:])    # scale
    
    DATA = data.iloc[:,1:]
    LABEL = data.iloc[:,0]
    
    # train data와 test data를 나누자.
    wisc_tr, wisc_ts, wisc_tr_lb, wisc_ts_lb = train_test_split(DATA,LABEL,test_size=0.2)
    
    # wisc_tr, wisc_tr_lb : training data, traing label
    # wisc_ts, wisc_ts_lb : test data, test label
    
    
                    # 검정과 예측
    model = KNeighborsClassifier(n_neighbors=k) 
    model.fit(wisc_tr, wisc_tr_lb)
    # model_prd = model.predict(wisc_ts) 예측 결과를 보고싶다면 주석을 해제
    model_score = model.score(wisc_ts,wisc_ts_lb)
    
    return model_score


