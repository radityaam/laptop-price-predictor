import pandas as pd
import streamlit as st
import numpy as np
import pickle

data = pd.read_csv("traineddata.csv")
file = open("pipe.pkl",  "rb")
pipe = pickle.load(file)
file.close()
st.title("Laptop price predictor")


company =  st.selectbox('Brand', data['Company'].unique())
laptop_type = st.selectbox('Type', data['TypeName'].unique())
ram = st.selectbox('RAM in GB', [2,4,6,8,12,16,24,32,64])
os = st.selectbox('OS', data['OpSys'].unique())
weight = st.number_input("Weight of the laptop")
touchscreen = st.selectbox('Touchscreen', ['No','Yes'])
ips = st.selectbox('IPS',  ['No','Yes'])
screen_size =  st.number_input('Screen Size: ')
resolution = st.selectbox('Sreen Resolution', ['1920x1080','1366x900',  '1600x900', '3840x2160','3200x1800','2880x1800','15560x1600','2560x1440','2304x1440',])
cpu = st.selectbox('CPU',data['CPU_Name'].unique())
hdd = st.selectbox('HDD in GB',[0,128,256,512,1024,2048])
ssd = st.selectbox('SSD in GB',[0,128,256,512,1024,2048])
gpu = st.selectbox('GPU',  data['gpu_brand'].unique())

if st.button('Predict Price'):
    ppi =None
    if touchscreen =='Yes':
        touchscreen  =1
    else:
        touchscreen = 0
    if ips == 'Yes':
        ips  = 1
    else:
        ips  = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])

    ppi =((X_res**2)+(Y_res**2))**0.5/(screen_size)

    query = np.array([company,laptop_type,ram,os,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu])

    query =query.reshape(1,12)

    prediction =int(np.exp(pipe.predict(query)[0]))

    st.title("Prediction price of the laptop is around" + str(prediction))