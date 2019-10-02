import os
import glob
import numpy as np
import pandas as pd

train = pd.read_csv("DataSet/Train.csv", nrows=1000)

total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
total_df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

print(list(total_df[total_df["Percent"] < 0.75]["Percent"].index))