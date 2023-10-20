import pandas as pd
import streamlit as st 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# page title
title = '<p style="font-family:sans-serif; font-weight: bold; color:#003366; font-size: 46px;">Bankruptcy Prevention Model Deployment</p>'
st.markdown(title, unsafe_allow_html=True)  

# sidebar title
sidebar_title = '<p style="font-family:sans-serif; font-weight: bold; color:#004d4d; font-size: 28px;">User Input Parameters</p>'
st.sidebar.markdown(sidebar_title, unsafe_allow_html=True) 

# sidebar user input parameter form
industrial_risk = st.sidebar.selectbox('industrial_risk',('1','0','0.5'))
management_risk = st.sidebar.selectbox('management_risk',('1','0','0.5'))
financial_flexibility = st.sidebar.selectbox('financial_flexibility',('1','0','0.5'))
credibility = st.sidebar.selectbox('credibilty',('1','0','0.5'))
competitiveness= st.sidebar.selectbox('competitiveness',('1','0','0.5'))
operating_risk= st.sidebar.selectbox('operating_risk',('1','0','0.5'))
   
data = {
    'industrial_risk':industrial_risk,
    ' management_risk':management_risk,
    ' financial_flexibility' : financial_flexibility,
    ' credibility': credibility,
    ' competitiveness':competitiveness,
    ' operating_risk':operating_risk,
}
user_input_parameters = pd.DataFrame(data,index = [0])

"""-----------------------------------------"""

user_input_header = '<p style="font-family:sans-serif; font-weight: bold; color:#004d4d; font-size: 28px;">User Input Parameters</p>'
st.markdown(user_input_header, unsafe_allow_html=True) 
st.write(user_input_parameters)

# Read CSV 
bankrupt= pd.read_csv("data/bankruptcy-prevention.csv",sep=";", header=0)

bankrupt_new = bankrupt.iloc[:,:]

bankrupt_new["class_yn"] = 1

bankrupt_new.loc[bankrupt[' class'] == 'bankruptcy', 'class_yn'] = 0

bankrupt_new.drop(' class', inplace = True, axis =1)

# Input
x = bankrupt_new.iloc[:,:-1]

# Target variable

y = bankrupt_new.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20, random_state = 0)

"""-----------------------------------------------------------------"""
col1, col2, col3 = st.columns(3)

with col1:
    # Model Building : Random Forest Classifier
    model_rf = RandomForestClassifier(n_estimators=10)
    model_rf=model_rf.fit(x_train, y_train)

    rf_pred = model_rf.predict(user_input_parameters)
    rf_prediction_proba = model_rf.predict_proba(user_input_parameters)

    rf_title = '<p style="font-family:sans-serif; font-weight: bold; color:#006600; font-size: 28px;">Random Forest Classifier</p>'
    st.markdown(rf_title, unsafe_allow_html=True) 

    predict_subheader = '<p style="font-family:sans-serif; font-weight: bold; color:black; font-size: 20px;">Predicted Result</p>'
    st.markdown(predict_subheader, unsafe_allow_html=True)  
    
    # Prediction Result Display
    st.write('Non Bankruptcy' if rf_prediction_proba[0][1] > 0.5 else 'Bankruptcy')

    predict_proba_subheader = '<p style="font-family:sans-serif; font-weight: bold; color:black; font-size: 20px;">Prediction Probability</p>'
    st.markdown(predict_proba_subheader, unsafe_allow_html=True) 

    # Prediction Probability Result Display
    st.write(rf_prediction_proba)

with col2:
    pass

with col3:
    # Model Building : Naive Bayes Classifier
    GNB = GaussianNB()
    Naive_GNB = GNB.fit(x_train ,y_train)

    prediction = GNB.predict(user_input_parameters)
    gnb_prediction_proba = GNB.predict_proba(user_input_parameters)

    gnb_title = '<p style="font-family:sans-serif; font-weight: bold; color:#006600; font-size: 28px;">Naive Bayes Classifier</p>'
    st.markdown(gnb_title, unsafe_allow_html=True)  
    st.markdown(predict_subheader, unsafe_allow_html=True)  
    
    st.write('Non Bankruptcy' if gnb_prediction_proba[0][1] > 0.5 else 'Bankruptcy')

    st.markdown(predict_proba_subheader, unsafe_allow_html=True)  
    st.write(gnb_prediction_proba)


"""-----------------------------------------"""




