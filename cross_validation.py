#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:43:03 2020

@author: Manting
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def K_fold_CV(k, data, label):
    #每一份的數量
    datasize = len(data) // k
    #Accuracy初始值
    accuracy = 0
    
    for i in range(k):
        X_test = data[i * datasize : (i + 1) * datasize]
        y_test = label[i * datasize : (i + 1) * datasize]
        X_train = np.concatenate( 
                         [data[ : i * datasize],
                         data[(i + 1) * datasize :]],
                         axis = 0)
        y_train = np.concatenate(
                         [label[ : i * datasize],
                         label[(i + 1) * datasize :]],
                         axis = 0)
    
        forest = RandomForestClassifier(n_estimators = 150,min_samples_split=12,
                             max_depth=20,oob_score=True,
                             random_state=42).fit(X_train, y_train)

        # 預測
        predictions = forest.predict(X_test)
        accuracy += accuracy_score(y_test, predictions)
        print(accuracy_score(y_test, predictions))
    return accuracy / k


# 讀資料
df = pd.read_csv('HW2data.csv')
# 將類別資料轉為整數，？值帶入最多的值
df["workclass"] = df["workclass"].replace(" ?", " Private").astype('category').cat.codes 
df["education"] = df["education"].astype('category').cat.codes 
df["marital_status"] = df["marital_status"].astype('category').cat.codes 
df["occupation"] = df["occupation"].replace(" ?", " Prof-specialty").astype('category').cat.codes 
df["relationship"] = df["relationship"].astype('category').cat.codes 
df["race"] = df["race"].astype('category').cat.codes 
df["sex"] = df["sex"].astype('category').cat.codes 
df["native_country"] = df["native_country"].replace(" ?", " United-States").astype('category').cat.codes 
df["income"] = df["income"].astype('category').cat.codes 

# 計算類別中出現最多次的值
#from collections import Counter
#word_counts = Counter(df["native_country"])
#print(word_counts)

X = df.drop("income", axis=1)
y = df["income"]

print("accuracy = ", K_fold_CV(10, X, y))






