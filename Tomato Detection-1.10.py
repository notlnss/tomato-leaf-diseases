import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image

# =========================================================
# KONFIGURASI
# =========================================================
MODEL_PATH = "final_best_model_tomato.pt"
IMG_SIZE = 128
CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]
NUM_CLASSES = len(CLASS_NAMES)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# ARSITEKTUR MODEL (harus sama persis dengan saat training)
# =========================================================
def build_transfer_model(num_classes=NUM_CLASSES):
    # weights=None karena kita akan load state_dict hasil training sendiri,
    # bukan bobot ImageNet lagi
    base_model = mobilenet_v2(weights=None)
    in_features = base_model.classifier[1].in_features  # 1280
    base_model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )
    return base_model


@st.cache_resource
def load_model():
    model = build_transfer_model(NUM_CLASSES)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


# =========================================================
# PREPROCESSING (harus sama persis dengan eval_resize + normalize saat training)
# =========================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def predict(image: Image.Image, model):
    img = image.convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    top_idx = int(torch.argmax(probs))
    return CLASS_NAMES[top_idx], probs.cpu().numpy()


# =========================================================
# UI STREAMLIT
# =========================================================
st.set_page_config(page_title="Klasifikasi Penyakit Daun Tomat", page_icon="🍅")
st.title("🍅 Klasifikasi Penyakit Daun Tomat")
st.write("Upload foto daun tomat untuk mendeteksi jenis penyakitnya (MobileNetV2 transfer learning).")

model = load_model()

uploaded_file = st.file_uploader("Pilih gambar daun tomat", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    with st.spinner("Memproses..."):
        label, probs = predict(image, model)

    label_display = label.replace("Tomato___", "").replace("_", " ")
    st.success(f"**Prediksi: {label_display}**")

    st.subheader("Detail Probabilitas")
    prob_dict = {
        c.replace("Tomato___", "").replace("_", " "): float(p)
        for c, p in zip(CLASS_NAMES, probs)
    }
    prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
    st.bar_chart(prob_dict)
