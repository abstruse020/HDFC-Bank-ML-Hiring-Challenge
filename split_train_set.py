import os
import glob
import numpy as np
import pandas as pd
import torch
import operator
import json
import gc
#from xgboost import XGBClassifier
import re, string
from fastai.tabular import *
import warnings

from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

gc.collect()

dict = {} 

def balance_dataset(dataset):
    print("----------------- balance dataset -----------------")
    print("Intial value of 0 and 1 ")
    print(dataset.loc[dataset['Col2'] == 0].shape[0], dataset.loc[dataset['Col2'] == 1].shape[0])
    smote = SMOTE()
    X, y = smote.fit_sample(dataset.drop(columns=['Col2']), dataset['Col2'])
    y = y[..., np.newaxis]
    Xy = np.concatenate((y, X), axis=1)
    df = pd.DataFrame.from_records(Xy, columns=list(dataset))
    df.Col2 = df.Col2.astype(int)
    print("After SMOTE value of 0 and 1 ")
    print(df.loc[df['Col2'] == 0].shape[0], df.loc[df['Col2'] == 1].shape[0])
    return df

def calculate_empty(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

def preprocess(dataset):
    for col in list(dataset.columns):
        if col != "Col1" and col != "Col2" and dataset[col].dtype == np.object_:
            dataset[col] = dataset[col].str.replace(u'\u2011', '-')
        if col != "Col1" and col != "Col2":
            dataset[col] = pd.to_numeric(dataset[col], errors="coerce")       
        
    return dataset

def preprocess_data(dataset):
    Col2_group = dataset.groupby(["Col2"])
    Col2 = Col2_group.median()
    for col in list(dataset.columns):
        if col != "Col1" and col != "Col2":
            dict.__setitem__(col, Col2[col].median())
    
    #Col2_transform = Col2_group.transform(lambda x: x.fillna(x.median()))
    #print(dict)
    for col in list(dataset.columns):
        if col != "Col1" and col != "Col2":
            #dataset[col] = Col2_transform[col]
            #dataset[col] = dataset[col].fillna(dict[col])
            dataset[col] = dataset[col].fillna(dataset[col].median())

    return dataset

def preprocess_data_test(dataset):
    for col in list(dataset.columns):
        if col != "Col1" and col != "Col2":
            dataset[col] = dataset[col].fillna(dataset[col].median())

    return dataset

def corelated_column():
    dataCorr = train.corr(method='pearson')
    dataCorr = dataCorr[abs(dataCorr) >= 0.01].stack().reset_index()
    dataCorr = dataCorr[dataCorr['level_0'].astype(str)!=dataCorr['level_1'].astype(str)]
    
    # filtering out lower/upper triangular duplicates 
    dataCorr['ordered-cols'] = dataCorr.apply(lambda x: '-'.join(sorted([x['level_0'],x['level_1']])),axis=1)
    dataCorr = dataCorr.drop_duplicates(['ordered-cols'])
    dataCorr.drop(['ordered-cols'], axis=1, inplace=True)
    dataCorr = dataCorr[dataCorr['level_0'] == "Col2"].sort_values(by=[0], ascending=False)

    print("-------------Writing columns-----------")
    csv_text = ",".join(dataCorr["level_1"].values)
    with open('columns.txt', 'w') as f:
        f.write(csv_text)

def remove_empty_column():
    total = train.isnull().sum().sort_values(ascending=False)
    percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
    total_df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    columns = list(total_df[total_df["Percent"] < 0.95]["Percent"].index)
    print("Columns with null < 15% ",len(columns))
    return columns


train = pd.DataFrame()
test = pd.DataFrame()
balanced_train = pd.DataFrame()
count = 1

for chunk in pd.read_csv("DataSet/Train.csv", iterator = True, chunksize=5000):
    print(chunk.shape)
    chunk = preprocess(chunk)
    # chunk.to_csv("DataSet/Train_{}.csv".format(count), index=False)
    count += 1
    #print(chunk.shape)
    train = pd.concat([train, chunk], ignore_index=True)

print("--------------------- Test -----------------------")

count = 1
for chunk in pd.read_csv("DataSet/Test.csv", iterator = True, chunksize=5000):
    print(chunk.shape)
    chunk = preprocess(chunk)
    # chunk.to_csv("DataSet/Train_{}.csv".format(count), index=False)
    count += 1
    test = pd.concat([test, chunk], ignore_index=True)

train_columns = remove_empty_column()
train_columns.remove("Col2")

test_columns = train_columns.copy()

train_columns.insert(0, "Col2")

train = preprocess_data(train[train_columns])
test = preprocess_data_test(test[test_columns])

train = train.drop(columns=['Col1'])

print(train.shape)
print(test.shape)

"""
prev = 0
for n in range(5000, train.shape[0], 5000):
    balanced_train = pd.concat([balanced_train, balance_dataset(train[prev:n])], ignore_index=True)
    prev = n
"""
print(train.shape)
print(test.shape)

#print(train.head())

print("--------------------- Train Null -----------------------")
print(print(calculate_empty(train).head(20)))

print("--------------------- Test Null -----------------------")
print(print(calculate_empty(test).head(20)))

train.to_csv("DataSet/Train_initial.csv", index=False)
test.to_csv("DataSet/Test_initial.csv", index=False)

print("--------------------- corelated_column -----------------------")
#corelated_column()