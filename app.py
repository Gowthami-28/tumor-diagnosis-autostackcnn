import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import gdown

st.title("Brain Tumor Detection App")
st.write("Upload an MRI image to get a diagnosis")


def load_models():
    cnn_path = 'custom_cnn_model.keras'
    if not os.path.exists(cnn_path):
        url = 'https://drive.google.com/uc?id=1Q42ZvWTrL-Vcp_pK7iHFolJyn9viNEku'
        gdown.download(url, cnn_path, quiet=False)
    
    custom_cnn = tf.keras.models.load_model(cnn_path)
    mobilenet = tf.keras.models.load_model('mobilenet_model.keras')
    return custom_cnn, mobilenet

custom_cnn, mobilenet = load_models()

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded MRI Image", width=600)
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred_cnn = custom_cnn.predict(img_array)
    pred_mobilenet = mobilenet.predict(img_array)
    
    stacked = (0.55 * pred_cnn + 0.45 * pred_mobilenet)
    
    if stacked[0][0] >= 0.5:
        st.error("Tumor Detected")
    else:
        st.success("No Tumor Detected")