# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 17:31:47 2020

@author: ChangYeol
"""

# 시각화를 통해서 k 값에 따른 score를 리턴하는 프로그램

def wisc_plt(k,n):
    # 입력한 k 값 범위까지 그래프를 그릴수 있습니다. ( 1  ~  k )
    
                    # import section
    import pandas as pd
    import numpy as np
    import matplotlib.pylab as plt
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
    wisc_tr, wisc_ts, wisc_tr_lb, wisc_ts_lb = train_test_split(DATA,LABEL,test_size=n)
    
    # wisc_tr, wisc_tr_lb : training data, traing label
    # wisc_ts, wisc_ts_lb : test data, test label
    
    
                    # 시각화하기
    result = pd.DataFrame()
    for i in range(1,k,2):               
        model = KNeighborsClassifier(n_neighbors=i) 
        model.fit(wisc_tr, wisc_tr_lb)
        model_score = model.score(wisc_ts,wisc_ts_lb)
        result = result.append({'k':i,'accuracy':model_score},ignore_index=True)
    
    plt.figure(figsize=(12,6))
    plt.plot(result['k'], result['accuracy'])
    plt.xticks(range(1,k,4),range(1,k,4))
    
    
wisc_plt(150,0.3)
