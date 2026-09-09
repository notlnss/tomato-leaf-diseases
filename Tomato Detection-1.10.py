import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import pandas as pd

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
LOW_CONFIDENCE_THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Info singkat + saran penanganan per kelas.
# Catatan: ini ringkasan umum untuk membantu interpretasi hasil model,
# bukan pengganti diagnosis dan rekomendasi dari ahli pertanian/pertanian setempat.
DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "deskripsi": "Infeksi bakteri yang menyebabkan bercak kecil kehitaman/kecoklatan pada daun, kadang dikelilingi halo kuning.",
        "saran": "Buang daun yang terinfeksi, hindari menyiram dari atas (daun basah mempercepat penyebaran), dan rotasi tanaman musim berikutnya.",
    },
    "Tomato___Early_blight": {
        "deskripsi": "Disebabkan jamur Alternaria, muncul sebagai bercak coklat konsentris ('target') mulai dari daun tua bagian bawah.",
        "saran": "Pangkas daun terinfeksi, perbaiki sirkulasi udara antar tanaman, dan pertimbangkan fungisida jika serangan meluas.",
    },
    "Tomato___Late_blight": {
        "deskripsi": "Penyakit serius akibat Phytophthora infestans, bercak coklat-kehitaman basah yang menyebar cepat terutama di cuaca lembap.",
        "saran": "Segera isolasi/buang tanaman terinfeksi, hindari kelembapan berlebih, dan gunakan fungisida sesuai anjuran — penyebarannya bisa sangat cepat.",
    },
    "Tomato___Leaf_Mold": {
        "deskripsi": "Jamur yang tumbuh di kondisi lembap tinggi, ditandai bercak kuning di permukaan atas daun dan lapisan berbulu di bawahnya.",
        "saran": "Kurangi kelembapan (ventilasi/greenhouse), hindari daun basah terlalu lama, dan buang daun yang terinfeksi berat.",
    },
    "Tomato___Septoria_leaf_spot": {
        "deskripsi": "Bercak kecil bulat dengan pusat abu-abu dan tepi gelap, biasanya dimulai dari daun bagian bawah.",
        "saran": "Buang daun terinfeksi, jaga jarak tanam untuk sirkulasi udara, dan hindari penyiraman yang membasahi daun.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "deskripsi": "Bukan penyakit jamur/bakteri, melainkan serangan hama tungau — daun tampak berbintik kuning pucat dan bisa muncul jaring halus.",
        "saran": "Semprot air bertekanan untuk merontokkan tungau, gunakan akarisida jika perlu, dan jaga kelembapan (tungau suka kondisi kering).",
    },
    "Tomato___Target_Spot": {
        "deskripsi": "Bercak coklat dengan pola cincin konsentris mirip early blight, disebabkan jamur Corynespora.",
        "saran": "Buang bagian tanaman terinfeksi, perbaiki drainase dan sirkulasi udara, pertimbangkan fungisida bila meluas.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "deskripsi": "Virus yang ditularkan kutu kebul (whitefly), menyebabkan daun menguning, menggulung ke atas, dan pertumbuhan kerdil.",
        "saran": "Kendalikan populasi kutu kebul, cabut dan musnahkan tanaman yang terinfeksi berat untuk mencegah penyebaran ke tanaman lain.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "deskripsi": "Virus yang menyebabkan pola belang hijau muda-tua (mosaik) pada daun serta daun keriput/mengecil.",
        "saran": "Tidak ada obat langsung — cabut tanaman terinfeksi, sterilkan alat berkebun, dan cuci tangan setelah kontak (virus mudah menular lewat sentuhan).",
    },
    "Tomato___healthy": {
        "deskripsi": "Daun tampak sehat, tidak menunjukkan tanda-tanda penyakit atau hama.",
        "saran": "Lanjutkan perawatan rutin: penyiraman teratur, pemupukan seimbang, dan pemantauan berkala.",
    },
}


# =========================================================
# ARSITEKTUR MODEL (harus sama persis dengan saat training)
# =========================================================
def build_transfer_model(num_classes=NUM_CLASSES):
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
    return probs.cpu().numpy()


def display_name(class_name: str) -> str:
    return class_name.replace("Tomato___", "").replace("_", " ")


def render_result(image: Image.Image, filename: str):
    st.image(image, caption=filename, use_container_width=True)

    with st.spinner("Memproses..."):
        probs = predict(image, model)

    order = probs.argsort()[::-1]
    top_class = CLASS_NAMES[order[0]]
    top_prob = float(probs[order[0]])

    if top_prob < LOW_CONFIDENCE_THRESHOLD:
        st.warning(
            f"Model kurang yakin (confidence {top_prob:.0%}). "
            "Coba foto dengan pencahayaan lebih baik, fokus tajam ke daun, dan latar belakang polos."
        )
    else:
        st.success(f"**Prediksi: {display_name(top_class)}**  ({top_prob:.0%})")

    info = DISEASE_INFO.get(top_class)
    if info:
        st.markdown(f"**Tentang kondisi ini:** {info['deskripsi']}")
        st.markdown(f"**Saran:** {info['saran']}")

    st.subheader("Top 3 Prediksi")
    for idx in order[:3]:
        st.write(f"{display_name(CLASS_NAMES[idx])} — {probs[idx]:.1%}")
        st.progress(float(probs[idx]))

    with st.expander("Lihat semua probabilitas kelas"):
        df = pd.DataFrame({
            "Kelas": [display_name(c) for c in CLASS_NAMES],
            "Probabilitas": probs,
        }).sort_values("Probabilitas", ascending=True)
        st.bar_chart(df.set_index("Kelas"), horizontal=True)


# =========================================================
# UI STREAMLIT
# =========================================================
st.set_page_config(page_title="Klasifikasi Penyakit Daun Tomat", page_icon="🍅", layout="centered")

with st.sidebar:
    st.header("ℹ️ Tentang Model")
    st.write("**Arsitektur:** MobileNetV2 (transfer learning)")
    st.write(f"**Jumlah kelas:** {NUM_CLASSES}")
    st.write(f"**Ukuran input:** {IMG_SIZE}×{IMG_SIZE} px")
    st.caption(
        "Hasil prediksi bersifat bantu-keputusan, bukan diagnosis final. "
        "Untuk kasus serius, konsultasikan dengan ahli pertanian/penyuluh setempat."
    )

st.title("🍅 Klasifikasi Penyakit Daun Tomat")
st.write("Upload atau ambil foto daun tomat untuk mendeteksi jenis penyakitnya.")

model = load_model()

tab_upload, tab_camera, tab_batch = st.tabs(["📁 Upload", "📷 Kamera", "🗂️ Upload Banyak Gambar"])

with tab_upload:
    uploaded_file = st.file_uploader("Pilih gambar daun tomat", type=["jpg", "jpeg", "png"], key="single")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        render_result(image, uploaded_file.name)

with tab_camera:
    camera_file = st.camera_input("Ambil foto daun tomat")
    if camera_file is not None:
        image = Image.open(camera_file)
        render_result(image, "Foto dari kamera")

with tab_batch:
    batch_files = st.file_uploader(
        "Pilih beberapa gambar sekaligus",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch",
    )
    if batch_files:
        for f in batch_files:
            st.divider()
            image = Image.open(f)
            render_result(image, f.name)
