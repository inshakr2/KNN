# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 16:24:11 2020

@author: ChangYeol
"""

# bmi file을 읽어 들이고 KNN을 통해 변수 값의 label을 예측해보기.
# 정규화와 표준화의 결과를 동시에 출력해보자.

def bmi(height, weight, k):
    # 입력값으로 키, 몸무게와 함께 k 값도 입력할 수 있습니다.

                        # import section
    import pandas as pd
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import scale
    from sklearn.preprocessing import minmax_scale
    
    
                        # data 추출 section
    data = pd.read_csv('c:/data/bmi.csv')
    
    X_train = np.array(data.iloc[:,:2])
    label = data.label
    
                        # 표준화 section
    # 표준화 학습 data set                        
    P_data = scale(X_train)
    # 표준화 예측 data
    P_h = (height - np.mean(data['height'])) / np.std(data['height'])
    P_w = (weight - np.mean(data['weight'])) / np.std(data['weight'])
    
    # 예측하기
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(P_data,label)
    P_res = model.predict(np.array([[P_h,P_w]]))
    
    
    
                        # 정규화 section
    # 정규화 학습 data set                        
    Z_data = minmax_scale(X_train)
    # 정규화 예측 data    
    Z_h = (height - np.min(data['height'])) / (np.max(data['height']) - np.min(data['height']))  
    Z_w = (weight - np.min(data['weight'])) / (np.max(data['weight']) - np.min(data['weight']))  
    
    # 예측하기
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(Z_data,label)
    Z_res = model.predict(np.array([[Z_h,Z_w]]))
    
    
                        # 출력
    print('표준화 결과 : ',P_res)
    print('정규화 결과 : ',Z_res)



