import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from io import BytesIO
import zipfile
import pandas as pd
import base64

# --- APP SETUP ---
st.set_page_config(
    page_title="NeuroNova | YOLO Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    body, .main {
        background-color: #0a0f1a;
        color: #d7d7d7;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    h1, h2, h3 {
        color: #00e5ff;
        text-shadow: 0 0 8px #00e5ff;
    }
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        border-radius: 12px;
        height: 2.8rem;
        font-weight: 600;
        box-shadow: 0 0 12px #00c6ff;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background: linear-gradient(90deg, #0072ff 0%, #00c6ff 100%);
        box-shadow: 0 0 18px #00c6ff;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    .uploaded-image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin-top: 1.5rem;
    }
    .image-card {
        background-color: #112233;
        padding: 0.8rem;
        border-radius: 16px;
        box-shadow: 0 0 12px #004080;
        text-align: center;
    }
    .image-card img {
        border-radius: 8px;
        max-width: 100%;
        height: auto;
    }
    .image-caption {
        margin-top: 0.5rem;
        color: #81d4fa;
        font-weight: 600;
        font-size: 0.9rem;
        word-wrap: break-word;
    }
    .remove-option {
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- UTILS ---
@st.cache_resource(show_spinner=False)
def load_model(path):
    return YOLO(path)

def draw_boxes(result, img):
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = box.conf[0]
        xyxy = box.xyxy[0].tolist()
        label = f"{result.names[cls_id]} {conf:.2f}"

        cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1]) - 20), (int(xyxy[0]) + w, int(xyxy[1])), (0, 255, 0), -1)
        cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return img

def convert_image_to_bytes(img):
    _, buf = cv2.imencode('.png', img)
    return buf.tobytes()

def create_zip(images_dict):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for filename, img_bytes in images_dict.items():
            zip_file.writestr(filename, img_bytes)
    return zip_buffer.getvalue()

def get_base64_download(data, filename):
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">⬇️</a>'
    return href

# --- SESSION STATE SETUP ---
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "annotated_images" not in st.session_state:
    st.session_state.annotated_images = {}
if "all_detections" not in st.session_state:
    st.session_state.all_detections = []

# --- SIDEBAR ---
st.sidebar.title("🛠️ Configuration")

weights_dir = Path.cwd()
model_files = [f.name for f in weights_dir.glob("*.pt")]

if not model_files:
    model_files = ["No models found"]

model_name = st.sidebar.selectbox("Select Model", model_files)

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

uploaded_files_new = st.sidebar.file_uploader(
    "📤 Upload Images (JPG/PNG)",
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=True
)

if uploaded_files_new:
    for file in uploaded_files_new:
        if file.name not in [f.name for f in st.session_state.uploaded_files]:
            st.session_state.uploaded_files.append(file)


st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 Developed by [Madhav Vashisht](https://madhavvashisht.unaux.com)")

# --- MAIN ---
st.title("🔭 NeuroNova | YOLO Detection")

if not st.session_state.uploaded_files:
    st.info("Upload images from the sidebar to begin detection.")
    st.stop()

model_path = weights_dir / model_name

try:
    model = load_model(str(model_path))
except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()

progress = st.progress(0)

if not st.session_state.annotated_images:
    st.session_state.annotated_images = {}
    st.session_state.all_detections = []

for i, file in enumerate(st.session_state.uploaded_files):
    if file.name in st.session_state.annotated_images:
        continue

    image = Image.open(file).convert("RGB")
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    preds = model.predict(img_cv, conf=confidence_threshold, verbose=False)
    result = preds[0]
    img_annot = draw_boxes(result, img_cv.copy())
    img_bytes = convert_image_to_bytes(img_annot)

    st.session_state.annotated_images[file.name] = img_bytes

    for box in result.boxes:
        st.session_state.all_detections.append({
            "Image": file.name,
            "Class": result.names[int(box.cls[0])],
            "Confidence": round(float(box.conf[0]), 3),
            "BBox": [round(float(coord), 2) for coord in box.xyxy[0].tolist()]
        })

    progress.progress((i + 1) / len(st.session_state.uploaded_files))

progress.empty()

# --- IMAGE OUTPUT GRID WITH FILTER ---
st.subheader("📸 Annotated Images")

filter_option = st.radio("Filter Images:", ["All", "Object Detected", "Object Not Detected"], horizontal=True)

# Determine which images to show
filtered_image_names = []
for img_name in st.session_state.annotated_images:
    has_detection = any(d["Image"] == img_name for d in st.session_state.all_detections)

    if filter_option == "All":
        filtered_image_names.append(img_name)
    elif filter_option == "Object Detected" and has_detection:
        filtered_image_names.append(img_name)
    elif filter_option == "Object Not Detected" and not has_detection:
        filtered_image_names.append(img_name)

# Display filtered images
if filtered_image_names:
    cols = st.columns(min(4, len(filtered_image_names)))
    for i, name in enumerate(filtered_image_names):
        with cols[i % len(cols)]:
            st.image(st.session_state.annotated_images[name], caption=name, use_container_width=True)
else:
    st.info(f"No images found for: {filter_option}")

# --- DETECTION TABLE + DOWNLOAD BUTTONS ---
st.subheader("🧾 Detection Results")

if st.session_state.all_detections:
    df = pd.DataFrame(st.session_state.all_detections)
    df["Download"] = df["Image"].apply(lambda name: get_base64_download(st.session_state.annotated_images[name], f"pred_{name}"))
    st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)

# --- ZIP DOWNLOAD ---
st.subheader("📦 Download All")
if len(st.session_state.annotated_images) > 1:
    zip_bytes = create_zip(st.session_state.annotated_images)
    st.download_button("⬇️ Download All as ZIP", zip_bytes, "yolo_predictions.zip", "application/zip")