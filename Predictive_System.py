# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 17:23:06 2022

@author: ssadh
"""

import numpy as np
import pickle
# loading the saved model
loaded_model = pickle.load(open('F:/Minor Project/trained_model.sav', 'rb'))

input_data=(3,550,4,2,7,8.05,8)
#changing the given data into numpy array 
input_data_as_numpy_array=np.asarray(input_data)
#reshaping the numpy array
input_data_reshape = input_data_as_numpy_array.reshape(1,-1)


prediction = loaded_model.predict(input_data_reshape)
print(prediction)
if prediction[0]==0:
    print("Dead")
else:
    print("Alive")