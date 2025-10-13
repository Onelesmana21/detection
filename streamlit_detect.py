import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

st.title("Simulasi Pixel Spot Detection - Object Detection")

uploaded_file = st.file_uploader("Unggah gambar di sini", type=["jpg","jpeg","png"])

# Load model deteksi objek pretrained YOLOv5 (misal)
@st.cache(allow_output_mutation=True)
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

model = load_model()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar Asli", use_container_width=True)

    # Prediksi objek pada gambar
    results = model(image)

    # Visualisasi bounding box dan label
    annotated_image = results.render()[0]
    annotated_image = Image.fromarray(annotated_image)

    st.image(annotated_image, caption="Hasil Deteksi Objek", use_container_width=True)

    st.write("Daftar objek terdeteksi:")
    df = results.pandas().xyxy[0]  # bounding box data
    st.dataframe(df[['name', 'confidence']])
else:
    st.info("Silakan unggah gambar untuk memulai deteksi.")

