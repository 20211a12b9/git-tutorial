# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.


"""
import numpy as np
import pandas as pd
import pickle
from xgboost import XGBRegressor

load_data=pickle.load(open('C:/Users/chowd/Downloads/crypto.sav','rb'))

input=(8,141.76,3,21.356,21.356,21.356,120.109)

input1 = np.asarray(input)

# reshape the np array as we are predicting for one instance
input_data_reshaped = input1.reshape(1,-1)

xbprediction = load_data.predict(input_data_reshaped)
print(xbprediction)


