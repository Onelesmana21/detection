import sys
sys.path.append('./yolov5')  # tambahkan path ke folder yolov5

import streamlit as st
from  import DetectMultiBackend
import torch
from PIL import Image
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    weights = 'models/yolov5n-seg.pt'  # sesuaikan path model
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

    # lakukan prediksi (adaptasi sesuai input model)
    results = model(img)

    # proses dan tampilkan hasil sesuai kebutuhan (misal bounding boxes)
    # ...
else:
    st.info("Unggah gambar untuk mulai deteksi")
