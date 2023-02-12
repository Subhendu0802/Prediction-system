# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 17:36:10 2022

@author: ssadh
"""
import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('F:/Minor Project/trained_model.sav', 'rb'))

#creating a function for prediction

def titanic_survival_prediction(input_data):

    #changing the given data into numpy array 
    input_data_as_numpy_array=np.asarray(input_data)
    #reshaping the numpy array
    input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshape)
    print(prediction)
    if prediction[0]==0:
        return 'Dead'
    else:
        return 'Alive'



def main():
    
    #giving a title
    st.title('Titanic Survival Prediction Web App')
    
    #getting the input data from user
    #Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
    
    Pclass = st.text_input('The Passenger Class')
    Sex = st.text_input('The Gender ')
    Age = st.text_input('The Passenger Age')
    SibSp = st.text_input('SibSp')
    Parch = st.text_input('Parch')
    Fare = st.text_input('Fare')
    Embarked = st.text_input('Embarked')
    
    
    #code for prediction
    prediction = ''
    
    #creating button for prediction
    if st.button('Predict'):
        prediction = st.titanic_survival_prediction([Pclass,Sex,Age,SibSp,Parch,Fare,Embarked])
        
        st.success(prediction)
    
if __name__ == '__main__':
    main()
