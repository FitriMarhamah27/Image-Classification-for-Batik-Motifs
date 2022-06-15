import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

file = st.file_uploader("Silahkan masukkan gambar tanaman", type=['jpg','png'])

def predict_stage(image_data,model):
    size = (150,150)
    image = ImageOps.fit(image_data,size, Image.ANTIALIAS)
    image_array = np.array(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
    data[0] = normalized_image_array
    preds = ""
    class_name =["Batik-Bali","Batik-Sogan","Batik-Kawung","Batik-Lasem","Batik-Garutan"]
    prediction = model.predict(data)
    confidence = round(100 * (np.max(prediction[0])), 2)
    final_pred = class_name[np.argmax(prediction)]
    return final_pred, confidence

if file is None:
    st.text("Silahkan masukkan file gambar")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    model = tf.keras.models.load_model('my_h5_model.h5')
    Generate_pred = st.button("Prediksi Gambar...")
    if Generate_pred:
        prediction = predict_stage(image, model)
        st.write(prediction)




    

  