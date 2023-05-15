import streamlit as st
import numpy as np
from pickle import load

scaler = load(open('C:\Users\Lenovo\Desktop\innomatics-proj\ablong-prod\model\model\standard_scaler.pkl', 'rb'))
knn = load(open('C:\Users\Lenovo\Desktop\innomatics-proj\ablong-prod\model\model\knn_model.pkl', 'rb'))

Sex = st.text_input("Sex", placeholder="Enter value ")
L1 = st.text_input("Length", placeholder="Enter value in cm")
L2 = st.text_input("diameter", placeholder="Enter value in cm")
L3 = st.text_input("height", placeholder="Enter value in cm")
L4 = st.text_input("whole weight", placeholder="Enter value in cm")
L5 = st.text_input("Shucked weight", placeholder="Enter value in cm")
L6 = st.text_input("viscera weight", placeholder="Enter value in cm")
L7 = st.text_input("shell weight", placeholder="Enter value in cm")

btn_click = st.button("Predict")

if btn_click == True:
    if Sex and L1 and L2 and L3 and L4 and L5 and L6 and L7:
        query_point = np.array([int(Sex), float(L1), float(L2), float(L3), float(L4), 
        float(L5), float(L6),float(L7)]).reshape(1, -1)
        query_point_transformed = scaler.transform(query_point)
        pred = knn.predict(query_point_transformed)
        st.success(pred)
    else:
        st.error("Enter the values properly.")