import streamlit as st
from yolov5 import YOLOv5
from PIL import Image

st.title("Simulasi Pixel Spot Detection - Object Detection")

uploaded_file = st.file_uploader("Unggah gambar di sini", type=["jpg", "jpeg", "png"])

@st.cache(allow_output_mutation=True)
def load_model():
    model_path = "yolov5n-seg.pt"  # Pastikan file ada di folder project root
    device = "cpu"  # Ganti ke "cuda" jika GPU tersedia
    model = YOLOv5(model_path, device)
    return model

model = load_model()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Asli", use_container_width=True)

    # Jalankan deteksi objek
    results = model.predict(image)

    # Tampilkan gambar hasil deteksi dengan bounding box
    annotated_img = results.render()[0]
    st.image(annotated_img, caption="Hasil Deteksi Objek", use_container_width=True)

    # Tampilkan dataframe objek terdeteksi
    df = results.pandas().xyxy[0]
    st.write("Daftar objek terdeteksi:")
    st.dataframe(df[['name', 'confidence']])
else:
    st.info("Silakan unggah gambar untuk memulai deteksi.")
