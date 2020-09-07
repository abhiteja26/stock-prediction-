import numpy as np
from numpy import *
import pandas as pd
from pandas import *
import matplotlib.pyplot as plt

dataset = pd.read_csv('JKP1.csv')

ds = dataset.set_index('Date')
x = ds.iloc[ :, :-2]
y = ds.iloc[:,-1]
import math

ds['Lag'] = ds.Lag.astype(float)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
out = pd.DataFrame({'actual':y_test,'predicted':y_pred})
print(out)

from sklearn.metrics import *
import math
from math import *
print('mean absolute error',mean_absolute_error(y_test,y_pred))
print('RMSE',sqrt(mean_squared_error(y_test,y_pred)))
