import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import joblib

# Baca dataset
dataset = 'students_adaptability_level_online_education.csv'
df = pd.read_csv(dataset)

st.subheader("Silakan Prediksi.")
# Dropdown untuk kolom "Gender"
Gender = st.selectbox('Input Gender', [i for i in sorted(df['Gender'].unique())])

# Dropdown untuk kolom "Age"
Age = st.selectbox('Input Age', [i for i in sorted(df['Age'].unique())])

# Dropdown untuk kolom "Education Level"
EducationLevel = st.selectbox('Input Education Level', [i for i in sorted(df['Education Level'].unique())])

# Dropdown untuk kolom "Institution Type"
InstitutionType = st.selectbox('Input Institution Type', [i for i in sorted(df['Institution Type'].unique())])

# Dropdown untuk kolom "IT Student"
ITStudent = st.selectbox('Input IT Student', [i for i in sorted(df['IT Student'].unique())])

# Dropdown untuk kolom "Location"
Location = st.selectbox('Input Location', [i for i in sorted(df['Location'].unique())])

# Dropdown untuk kolom "Load-shedding"
LoadShedding = st.selectbox('Input Load-shedding', [i for i in sorted(df['Load-shedding'].unique())])

# Dropdown untuk kolom "Financial Condition"
FinancialCondition = st.selectbox('Input Financial Condition', [i for i in sorted(df['Financial Condition'].unique())])

# Dropdown untuk kolom "Internet Type"
InternetType = st.selectbox('Input Internet Type', [i for i in sorted(df['Internet Type'].unique())])

# Dropdown untuk kolom "Network Type"
NetworkType = st.selectbox('Input Network Type', [i for i in sorted(df['Network Type'].unique())])

# Dropdown untuk kolom "Class Duration"
ClassDuration = st.selectbox('Input Class Duration', [i for i in sorted(df['Class Duration'].unique())])

# Dropdown untuk kolom "Self Lms"
SelfLms = st.selectbox('Input Self Lms', [i for i in sorted(df['Self Lms'].unique())])

# Dropdown untuk kolom "Device"
Device = st.selectbox('Input Device', [i for i in sorted(df['Device'].unique())])

data = pd.DataFrame({
    'Gender': [df[df['Gender'] == Gender].index[0]],
    'Age': [df[df['Age'] == Age].index[0]],
    'Education Level': [df[df['Education Level'] == EducationLevel].index[0]],
    'Institution Type': [df[df['Institution Type'] == InstitutionType].index[0]],
    'IT Student': [df[df['IT Student'] == ITStudent].index[0]],
    'Location': [df[df['Location'] == Location].index[0]],
    'Load-shedding': [df[df['Load-shedding'] == LoadShedding].index[0]],
    'Financial Condition': [df[df['Financial Condition'] == FinancialCondition].index[0]],
    'Internet Type': [df[df['Internet Type'] == InternetType].index[0]],
    'Network Type': [df[df['Network Type'] == NetworkType].index[0]],
    'Class Duration': [df[df['Class Duration'] == ClassDuration].index[0]],
    'Self Lms': [df[df['Self Lms'] == SelfLms].index[0]],
    'Device': [df[df['Device'] == Device].index[0]]
})
button = st.button('Predict')

if button:

    # filename='dtc.pkl'
    # with open(filename,'rb') as file:
    #     loaded_model = pickle.load(file)

    # with open ('knnnew.pkl','rb') as file:
    #     loaded_model = pickle.load(file)

    loaded_model = joblib.load('dtcj.pkl')

    predicted = loaded_model.predict(data)
    
    if predicted[0] == 0:
        st.write('High')
    elif predicted[0] == 1:
        st.write('Low')
    elif predicted[0] == 2:
        st.write('Moderate')
    else:
        st.write('Not Defined')