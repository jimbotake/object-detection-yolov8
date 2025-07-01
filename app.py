import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # Gunakan model ringan: yolov8n.pt

model = load_model()

st.title("🎯 Deteksi Objek dengan YOLOv8")
st.write("Upload gambar untuk mendeteksi objek menggunakan model YOLOv8.")

uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    with st.spinner("Mendeteksi objek..."):
        results = model(image)  # Deteksi
        result_image = results[0].plot()  # Gambar hasil deteksi

    st.image(result_image, caption="Hasil Deteksi", use_column_width=True)
