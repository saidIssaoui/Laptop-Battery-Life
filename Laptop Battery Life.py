import math
import os
import random
import re
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



data = pd.read_csv('trainingdata.txt',sep=',')


max_value = data.iloc[:, 1].max()
data2 = data[data.iloc[:, 1] == max_value]
data3 = data[data.iloc[:, 1] < max_value]

min_value = data2.iloc[:, 0].min()


x= data3.iloc[:, 0].values.reshape(45, 1)
y= data3.iloc[:, 1].values.reshape(45, 1)

xtrain, xtest, ytrain, ytest = train_test_split(x,y,random_state=1)
model = LinearRegression().fit(xtrain,ytrain)


#print('how much you charge the battery :')
cb = float(input())
if cb >= min_value :
   print(max_value)
else :
    y_pred = model.predict([[cb]])
    c = float(y_pred)
    print('',c)