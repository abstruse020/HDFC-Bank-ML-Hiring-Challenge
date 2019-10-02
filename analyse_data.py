import os
import glob
import numpy as np
import pandas as pd
import torch
import operator
import json
from imblearn.over_sampling import SMOTE
#from xgboost import XGBClassifier
import re, string


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("DataSet/Train_initial.csv", nrows=200)

print(train.loc[train['Col2'] == 0].shape[0], train.loc[train['Col2'] == 1].shape[0])

smote = SMOTE()
X, y = smote.fit_sample(train.drop(columns=['Col2']), train['Col2'])
y = y[..., np.newaxis]
Xy = np.concatenate((y, X), axis=1)
df = pd.DataFrame.from_records(Xy, columns=list(train))
#df.Col2 = df.Col2.astype(int)
print(df.head())

"""
# train = train[:1000]
k = 50
corrmat = train.corr()
ser_corr = corrmat.nlargest(k, 'Col2')['Col2']
ser_cols = ser_corr.index
transpose = train[ser_cols].values.T
transpose = transpose.astype(float)

cm = np.corrcoef(transpose)
#.astype(float)
#plt.subplots(figsize=(26, 26))
#sns.set(font_scale=0.75)
sns.heatmap(cm, cbar=True, annot=True, square=True)
"""


plt.figure(figsize=(6, 26))
sns.countplot(x="Col2", data=df)
plt.show()

