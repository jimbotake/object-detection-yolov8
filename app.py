import streamlit as st
from PIL import Image
import torch

@st.cache_resource
def load_model():
    # Gunakan YOLOv5 versi 6.2 agar tidak butuh ultralytics
    return torch.hub.load('ultralytics/yolov5:v6.2', 'yolov5s', pretrained=True)

model = load_model()

st.title("🎯 Object Detection dengan YOLOv5")
st.write("Upload gambar untuk mendeteksi objek.")

uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # Konversi agar tidak error pada PNG
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    with st.spinner("Mendeteksi objek..."):
        results = model(image)
        results.render()  # menggambar box ke dalam image

    st.image(results.ims[0], caption="Hasil Deteksi", use_column_width=True)
