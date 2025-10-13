import sys
sys.path.append('./yolov5')  # folder yolov5 ada di root repo

import streamlit as st
from models.common import DetectMultiBackend
import torch
from PIL import Image
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    weights = 'yolov5n-seg.pt'  # path model relatif terhadap root
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend(weights, device=device)
    model.eval()
    return model

model = load_model()

uploaded_file = st.file_uploader("Unggah gambar", type=['jpg','jpeg','png'])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar Asli", use_container_width=True)
    img = np.array(image)

    results = model(img)  # pastikan adaptasi panggilan sesuai dengan API DetectMultiBackend
    # proses hasil dan tampilkan sesuai kebutuhan
else:
    st.info("Unggah gambar untuk mulai deteksi")
