import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import io
from model import SimpleCNN

# Cấu hình
classes = ["damaged", "non_damaged"]
image_size = 112
model_path = "trained_models/best_cnn.pt"

# Load model
@st.cache_resource
def load_model():
    model = SimpleCNN(num_classes=2)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model

model = load_model()
softmax = nn.Softmax(dim=1)

# Giao diện
st.title("🌾 Crop Health Classifier")
st.write("Tải lên ảnh cây trồng để kiểm tra tình trạng: **Damaged** hoặc **Non-Damaged**")

uploaded_file = st.file_uploader("Chọn một ảnh...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh đã tải lên", use_container_width=True)

    # Tiền xử lý
    image_resized = image.resize((image_size, image_size))
    image_np = np.array(image_resized).astype(np.float32) / 255.0
    image_np = np.transpose(image_np, (2, 0, 1))  # CHW
    image_tensor = torch.tensor(image_np).unsqueeze(0)  # B x C x H x W

    # Dự đoán
    with torch.no_grad():
        output = model(image_tensor)
        probs = softmax(output)
        confidence, predicted_idx = torch.max(probs, 1)
        predicted_label = classes[predicted_idx.item()]
        confidence_score = confidence.item()

    # Hiển thị kết quả
    st.subheader("🧠 Kết quả phân loại:")
    st.success(f"**{predicted_label.upper()}** ({confidence_score * 100:.2f}%)")
