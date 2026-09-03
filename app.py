import streamlit as st
import pandas as pd
import pickle

# lodel binary code
with open("model_pipe.pkl","rb")as file:
    model=pickle.load(file)

st.title("Credit Card Default Risk Prediction System")

# input features 


LIMIT_BAL=st.number_input('LIMIT_BAL',min_value=0 )
PAY_0=st.number_input('PAY_0')
PAY_2=st.number_input('PAY_2')
PAY_3=st.number_input('PAY_3')
PAY_4=st.number_input('PAY_4')
PAY_5=st.number_input('PAY_5')       
PAY_6=st.number_input('PAY_6')
        
BILL_AMT1=st.number_input('BILL_AMT1')
BILL_AMT2=st.number_input('BILL_AMT2')
# BILL_AMT3=st.number_input('BILL_AMT3')
# BILL_AMT4=st.number_input('BILL_AMT4')
# BILL_AMT5=st.number_input('BILL_AMT5')
# BILL_AMT6=st.number_input('BILL_AMT6')

PAY_AMT1=st.number_input('PAY_AMT1') 
PAY_AMT2=st.number_input('PAY_AMT2') 
PAY_AMT3=st.number_input('PAY_AMT3') 
PAY_AMT4=st.number_input('PAY_AMT4') 
PAY_AMT5=st.number_input('PAY_AMT5') 
PAY_AMT6=st.number_input('PAY_AMT6') 
      

# input_data storing all the entries in to data frame
import pandas as pd

input_data = pd.DataFrame({
    'LIMIT_BAL': [LIMIT_BAL],

    'PAY_0': [PAY_0],
    'PAY_2': [PAY_2],
    'PAY_3': [PAY_3],
    'PAY_4': [PAY_4],
    'PAY_5': [PAY_5],
    'PAY_6': [PAY_6],

    'BILL_AMT1': [BILL_AMT1],
    'BILL_AMT2': [BILL_AMT2],
    # 'BILL_AMT3': [BILL_AMT3],
    # 'BILL_AMT4': [BILL_AMT4],
    # 'BILL_AMT5': [BILL_AMT5],
    # 'BILL_AMT6': [BILL_AMT6],

    'PAY_AMT1': [PAY_AMT1],
    'PAY_AMT2': [PAY_AMT2],
    'PAY_AMT3': [PAY_AMT3],
    'PAY_AMT4': [PAY_AMT4],
    'PAY_AMT5': [PAY_AMT5],
    'PAY_AMT6': [PAY_AMT6]
})


if st.button('Predict'):
    prediction=model.predict(input_data)

    if prediction[0] in [1,"Y"]:
        st.write("Default")
    else:
        st.write('No Default')
