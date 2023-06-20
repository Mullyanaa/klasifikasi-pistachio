import pickle
import streamlit as st
import numpy as np

# membaca model
model = pickle.load(open('pistachio.sav', 'rb'))
scaler = pickle.load(open('scaler.sav','rb'))

#judul web
st.title('Prediksi Jenis Pistachio')

#membagi kolom
col1, col2 = st.columns(2)

with col1 :
    AREA = st.number_input('input nilai luas pistachio')
    PERIMETER = st.number_input('input nilai keliling pistachio')
    MAJOR_AXIS = st.number_input('input nilai sumbu utama pistachio')
    MINOR_AXIS = st.number_input('input nilai sumbu minor pistachio')
    ECCENTRICITY = st.number_input('input nilai eksentrisitas pistachio')
    EQDIASQ = st.number_input('input nilai luas sama dengan diameter persegi pistachio')

with col2 :
    SOLIDITY = st.number_input('input nilai kepadatan pistachio')
    CONVEX_AREA = st.number_input('input nilai luas cembung pistachio')
    EXTENT = st.number_input('input nilai tingkat perluasan pistachio')
    ASPECT_RATIO = st.number_input('input nilai rasio aspek pistachio')
    ROUNDNESS = st.number_input('input nilai kebulatan pistachio')
    COMPACTNESS = st.number_input('input nilai kekompakan pistachio')

# code untuk prediksi
prediction = ''
input_data = (AREA,PERIMETER,MAJOR_AXIS,MINOR_AXIS,ECCENTRICITY,EQDIASQ,SOLIDITY,
              CONVEX_AREA,EXTENT,ASPECT_RATIO,ROUNDNESS,COMPACTNESS)

input_data_as_numpy_array = np.array(input_data)

input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

std_data = scaler.transform(input_data_reshape)

# membuat tombol untuk prediksi
if st.button('Proses'):
    pistachio_prediction = model.predict(std_data)
    if(pistachio_prediction[0] == 1):
        prediction = 'Jenis Kirmizi Pistachio'
    else:
        prediction = 'Jenis Siit Pistachio'
    st.success(prediction)
