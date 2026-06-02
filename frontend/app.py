"""
VisionForge - AI Image Classifier (Production Ready)
Frontend for Streamlit Cloud → Hugging Face Backend
"""
import streamlit as st
import requests
import io
import json
import time
from PIL import Image
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ✅ PRODUCTION: Hugging Face Backend URL
API = "https://Zay7ab-visionforge-backend.hf.space"

st.set_page_config(
    page_title="VisionForge",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Modern CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

* { font-family: 'Inter', sans-serif; }

.main { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }

.feature-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 2rem; text-align: center;
    transition: all 0.3s ease; height: 100%;
}
.feature-card:hover {
    transform: translateY(-10px);
    box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
    border: 1px solid rgba(102, 126, 234, 0.5);
}

.hero-section {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 30px; padding: 3rem;
    text-align: center; margin-bottom: 2rem;
}

.hero-title {
    font-size: 3.5rem; font-weight: 900;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
}

.stat-container {
    display: flex; justify-content: center; gap: 3rem; margin: 2rem 0;
}
.stat-number {
    font-size: 2.5rem; font-weight: 900;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-label {
    font-size: 0.9rem; color: rgba(255, 255, 255, 0.6);
    text-transform: uppercase; letter-spacing: 2px;
}

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 600 !important;
    padding: 0.8rem 2rem !important; font-size: 1rem !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4) !important;
}
.stButton > button:disabled {
    opacity: 0.4 !important;
    transform: none !important;
    box-shadow: none !important;
}

.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.5), transparent);
    margin: 2rem 0;
}

.info-box {
    background: rgba(255, 255, 255, 0.03);
    border-left: 4px solid #667eea;
    border-radius: 10px; padding: 1.5rem; margin: 1rem 0;
}

section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.8) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
}

.stTabs [data-baseweb="tab"] { color: rgba(255,255,255,0.6) !important; }
.stTabs [aria-selected="true"] { color: #667eea !important; }
.stSlider > div { color: white !important; }
.stTextInput > div > input { background: rgba(255,255,255,0.1) !important; color: white !important; border: 1px solid rgba(255,255,255,0.2) !important; }
.stSelectbox > div > div { background: rgba(255,255,255,0.1) !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────
defaults = {
    "classes_confirmed": False,
    "trained": False,
    "accuracy": None,
    "class_names": [],
    "img_counts": {},
    "uploaded_tracker": {},
    "training_history": [],
    "confusion_matrix": [],
    "prediction_history": [],
    "active_tab": "🏠 Home",
    "selected_model_name": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── API Helpers ───────────────────────────────────────────────────────────
def api(method, path, **kwargs):
    try:
        r = getattr(requests, method)(f"{API}{path}", timeout=120, **kwargs)
        if r.ok:
            return r.json()
        return None
    except:
        return None

def api_upload(cls, img_bytes, fname):
    try:
        r = requests.post(
            f"{API}/upload/{requests.utils.quote(cls, safe='')}",
            files={"file": (fname, img_bytes, "image/jpeg")},
            timeout=20,
        )
        return r.json().get("count") if r.ok else None
    except:
        return None

def api_predict(img_bytes):
    try:
        r = requests.post(
            f"{API}/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            timeout=20,
        )
        return r.json() if r.ok else None
    except:
        return None

def pil_to_bytes(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="font-size: 2rem; margin-bottom: 0.2rem; color: white;">🧠 VisionForge</h1>
        <p style="color: rgba(255,255,255,0.5); font-size: 0.8rem;">AI Image Classifier</p>
    </div>
    """, unsafe_allow_html=True)
    
    status = api("get", "/status")
    if status:
        st.success("🟢 Backend Online")
    else:
        st.error("🔴 Backend Offline")
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    st.markdown("### 📍 Navigation")
    pages = ["🏠 Home", "🎓 Train", "🔍 Predict", "📊 Analytics", "💾 Models", "⚙️ Settings"]
    for page in pages:
        if st.button(page, use_container_width=True, key=f"nav_{page}"):
            st.session_state.active_tab = page
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    if st.session_state.trained:
        st.markdown("### 📈 Active Model")
        st.metric("Accuracy", f"{st.session_state.accuracy}%")
        st.metric("Classes", len(st.session_state.class_names))
        if st.session_state.selected_model_name:
            st.caption(f"Model: {st.session_state.selected_model_name}")
        if st.session_state.class_names:
            st.caption(f"Classes: {', '.join(st.session_state.class_names)}")
    
    if st.session_state.classes_confirmed:
        st.markdown("### 📊 Dataset")
        for cls in st.session_state.class_names:
            cnt = st.session_state.img_counts.get(cls, 0)
            st.caption(f"{cls}: {cnt} images")
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    if st.button("🗑️ Reset All", use_container_width=True):
        api("delete", "/reset")
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
    
    st.caption("v4.0 | MobileNetV3 | Made with ❤️")


# ═══════════════════════════════════════════════════════════════════════════
# 🏠 HOME PAGE
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.active_tab == "🏠 Home":
    
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🧠 VisionForge</h1>
        <p style="font-size: 1.2rem; color: rgba(255, 255, 255, 0.7); margin-bottom: 2rem;">Train custom AI models to recognize anything. No coding required.</p>
        <div class="stat-container">
            <div class="stat-item"><div class="stat-number">⚡</div><div class="stat-label">Real-Time</div></div>
            <div class="stat-item"><div class="stat-number">🎯</div><div class="stat-label">95%+ Accuracy</div></div>
            <div class="stat-item"><div class="stat-number">📱</div><div class="stat-label">Any Device</div></div>
            <div class="stat-item"><div class="stat-number">🔒</div><div class="stat-label">100% Private</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🚀 Start Training", use_container_width=True, key="home_train"):
            st.session_state.active_tab = "🎓 Train"
            st.rerun()
    with col2:
        if st.button("🔍 Test Model", use_container_width=True, key="home_test"):
            st.session_state.active_tab = "🔍 Predict"
            st.rerun()
    with col3:
        if st.button("📊 View Analytics", use_container_width=True, key="home_analytics"):
            st.session_state.active_tab = "📊 Analytics"
            st.rerun()
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    st.markdown("### 🎯 Core Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="feature-card"><div style="font-size:3rem;">🧠</div><div style="color:white;font-weight:700;font-size:1.2rem;">Transfer Learning</div><p style="color:rgba(255,255,255,0.6);">Powered by MobileNetV3 for state-of-the-art accuracy with minimal data</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="feature-card"><div style="font-size:3rem;">⚡</div><div style="color:white;font-weight:700;font-size:1.2rem;">Real-Time Prediction</div><p style="color:rgba(255,255,255,0.6);">Instant predictions with confidence scores via webcam or upload</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="feature-card"><div style="font-size:3rem;">📊</div><div style="color:white;font-weight:700;font-size:1.2rem;">Advanced Analytics</div><p style="color:rgba(255,255,255,0.6);">Confusion matrices, training curves, detailed metrics</p></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="feature-card"><div style="font-size:3rem;">🎨</div><div style="color:white;font-weight:700;font-size:1.2rem;">Data Augmentation</div><p style="color:rgba(255,255,255,0.6);">Auto-generate variations: flips, rotations, brightness</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="feature-card"><div style="font-size:3rem;">💾</div><div style="color:white;font-weight:700;font-size:1.2rem;">Model Management</div><p style="color:rgba(255,255,255,0.6);">Save, load, export models for deployment anywhere</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="feature-card"><div style="font-size:3rem;">🔒</div><div style="color:white;font-weight:700;font-size:1.2rem;">100% Private</div><p style="color:rgba(255,255,255,0.6);">All processing local. Data never leaves your machine</p></div>', unsafe_allow_html=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    st.markdown("### 🚀 How It Works")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div style="text-align:center;padding:1rem;"><div style="font-size:3rem;">1️⃣</div><h4 style="color:white;">Define Classes</h4><p style="color:rgba(255,255,255,0.6);">Name what to recognize</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div style="text-align:center;padding:1rem;"><div style="font-size:3rem;">2️⃣</div><h4 style="color:white;">Upload Images</h4><p style="color:rgba(255,255,255,0.6);">5-10 examples per class</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div style="text-align:center;padding:1rem;"><div style="font-size:3rem;">3️⃣</div><h4 style="color:white;">Train Model</h4><p style="color:rgba(255,255,255,0.6);">AI learns patterns</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div style="text-align:center;padding:1rem;"><div style="font-size:3rem;">4️⃣</div><h4 style="color:white;">Predict</h4><p style="color:rgba(255,255,255,0.6);">Real-time classification</p></div>', unsafe_allow_html=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="info-box"><h4 style="color:#667eea;">🎯 Why VisionForge?</h4><ul style="color:rgba(255,255,255,0.7);"><li>Transfer Learning with MobileNetV3</li><li>Works with 5-10 images per class</li><li>CPU & GPU auto-detection</li><li>Export models for production</li></ul></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="info-box"><h4 style="color:#667eea;">💡 Use Cases</h4><ul style="color:rgba(255,255,255,0.7);"><li>Product recognition</li><li>Plant disease detection</li><li>Gesture recognition</li><li>Quality control</li></ul></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# 🎓 TRAIN PAGE
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.active_tab == "🎓 Train":
    st.markdown("# 🎓 Train Your Model")
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    done1 = st.session_state.classes_confirmed
    
    if not done1:
        st.markdown("### Step 1: Define Classes")
        num_cls = st.number_input("Number of classes", 2, 8, 2)
        cols = st.columns(min(int(num_cls), 4))
        names = []
        for i in range(int(num_cls)):
            with cols[i % 4]:
                n = st.text_input(f"Class {i+1}", value=f"Class {i+1}", key=f"cn_{i}")
                names.append(n.strip())
        
        if st.button("✅ Confirm Classes", use_container_width=True):
            unique = list(dict.fromkeys([n for n in names if n]))
            if len(unique) < 2:
                st.error("Need at least 2 different class names.")
            elif not status:
                st.error("Backend not running. Start backend first!")
            else:
                result = api("post", "/classes", json={"classes": unique})
                if result:
                    st.session_state.class_names = unique
                    st.session_state.img_counts = {c: 0 for c in unique}
                    st.session_state.uploaded_tracker = {c: set() for c in unique}
                    st.session_state.classes_confirmed = True
                    st.rerun()
    else:
        st.success(f"Classes: **{'  ·  '.join(st.session_state.class_names)}**")
        if st.button("✏️ Edit Classes"):
            st.session_state.classes_confirmed = False
            st.session_state.trained = False
            st.rerun()
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    if st.session_state.classes_confirmed:
        classes = st.session_state.class_names
        all_ready = all(st.session_state.img_counts.get(c, 0) >= 2 for c in classes)
        
        st.markdown("### Step 2: Upload Training Images")
        st.caption("Upload at least 3 images per class for best results")
        
        tabs = st.tabs([f"  {c}  " for c in classes])
        for i, cls in enumerate(classes):
            with tabs[i]:
                cnt = st.session_state.img_counts.get(cls, 0)
                st.metric(f"Images for {cls}", cnt)
                
                uploaded = st.file_uploader(
                    f"Upload images",
                    type=["jpg","jpeg","png","webp","bmp"],
                    accept_multiple_files=True,
                    key=f"up_{cls}",
                )
                
                if cls not in st.session_state.uploaded_tracker:
                    st.session_state.uploaded_tracker[cls] = set()
                
                if uploaded:
                    new = [f for f in uploaded if f.name not in st.session_state.uploaded_tracker[cls]]
                    if new:
                        bar = st.progress(0, text="Uploading...")
                        ok_count = 0
                        for j, uf in enumerate(new):
                            try:
                                img_bytes = pil_to_bytes(Image.open(uf).convert("RGB"))
                                result = api_upload(cls, img_bytes, uf.name)
                                if result is not None:
                                    st.session_state.img_counts[cls] = result
                                    st.session_state.uploaded_tracker[cls].add(uf.name)
                                    ok_count += 1
                            except:
                                pass
                            bar.progress((j+1)/len(new), text=f"Uploading {j+1}/{len(new)}...")
                        bar.empty()
                        if ok_count:
                            st.success(f"✅ {ok_count} images added to **{cls}**")
                            st.rerun()
        
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        
        st.markdown("### Step 3: Train Model")
        
        if not all_ready:
            missing = [c for c in classes if st.session_state.img_counts.get(c, 0) < 2]
            st.warning(f"Need more images for: {', '.join(missing)}")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🚀 Start Training", use_container_width=True, disabled=st.session_state.trained):
                with st.spinner("Training with MobileNetV3... This may take a minute ⚙️"):
                    result = api("post", "/train")
                if result:
                    st.session_state.trained = True
                    st.session_state.accuracy = result.get("accuracy", 0)
                    st.session_state.training_history = result.get("history", [])
                    st.session_state.confusion_matrix = result.get("confusion_matrix", [])
                    st.session_state.selected_model_name = None
                    st.success(f"✅ Training complete! Accuracy: **{result['accuracy']}%**")
                    st.rerun()
                else:
                    st.error("Training failed. Check that each class has images.")
        
        if st.session_state.trained and st.session_state.accuracy:
            with col2:
                st.metric("Model Accuracy", f"{st.session_state.accuracy}%")
        
        if st.session_state.trained:
            st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Retrain Model", use_container_width=True):
                    st.session_state.trained = False
                    st.session_state.accuracy = None
                    st.session_state.training_history = []
                    st.session_state.confusion_matrix = []
                    st.session_state.selected_model_name = None
                    st.rerun()
            with col2:
                if st.button("🗑️ Reset & Start Fresh", use_container_width=True):
                    api("delete", "/reset")
                    for k in list(st.session_state.keys()):
                        del st.session_state[k]
                    st.rerun()
        
        if st.session_state.trained:
            st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
            st.markdown("### 💾 Save Model")
            col1, col2 = st.columns([3, 1])
            with col1:
                model_name = st.text_input("Model name", value=f"model_{datetime.now().strftime('%H%M')}", key="save_name")
            with col2:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                if st.button("💾 Save", use_container_width=True):
                    r = api("post", "/model/save", json={"name": model_name})
                    if r:
                        st.success(f"✅ Model saved as **{model_name}**")
                    else:
                        st.error("Save failed.")


# ═══════════════════════════════════════════════════════════════════════════
# 🔍 PREDICT PAGE (With Model Selection + Camera Fix)
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.active_tab == "🔍 Predict":
    st.markdown("# 🔍 Predict")
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    st.markdown("### 📂 Select Model for Prediction")
    
    result = api("get", "/model/list")
    saved_models = result.get("models", []) if result else []
    
    model_options = []
    
    if st.session_state.trained:
        current_label = f"📌 Current Trained Model | Accuracy: {st.session_state.accuracy}% | Classes: {', '.join(st.session_state.class_names)}"
        model_options.append(current_label)
    
    for m in saved_models:
        saved_label = f"💾 {m['name']} | Accuracy: {round(m['accuracy']*100,1)}% | Classes: {', '.join(m['classes'])}"
        model_options.append(saved_label)
    
    if not model_options:
        st.warning("⚠️ No models available! Please train a model first from the **🎓 Train** tab.")
    else:
        selected = st.selectbox(
            "Choose which model to use:",
            model_options,
            key="model_selector"
        )
        
        if selected.startswith("💾"):
            model_name = selected.split(" | ")[0].replace("💾 ", "")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("📂 Load Model", use_container_width=True):
                    r = api("post", f"/model/load/{model_name}")
                    if r:
                        st.session_state.trained = True
                        st.session_state.accuracy = r.get("accuracy", 0)
                        st.session_state.class_names = r.get("classes", [])
                        st.session_state.classes_confirmed = True
                        st.session_state.img_counts = {c: 0 for c in r.get("classes", [])}
                        st.session_state.selected_model_name = model_name
                        st.session_state.training_history = r.get("history", [])
                        st.session_state.confusion_matrix = r.get("confusion_matrix", [])
                        st.success(f"✅ Model **{model_name}** loaded successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to load model.")
            with col2:
                st.info("👆 Click 'Load Model' to activate this saved model for prediction")
        else:
            st.session_state.selected_model_name = None
            st.success("✅ Using current trained model - Ready for prediction!")
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    if st.session_state.trained:
        st.markdown("### 📤 Upload Image to Predict")
        st.caption(f"Active Model Classes: **{', '.join(st.session_state.class_names)}**")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📁 Input Source")
            mode = st.radio("Choose input:", ["📁 Upload Image", "📷 Camera"], horizontal=True, key="test_mode")
            test_bytes = None
            
            if mode == "📁 Upload Image":
                f = st.file_uploader("Select an image", type=["jpg","jpeg","png","webp"], key="test_file")
                if f:
                    img = Image.open(f).convert("RGB")
                    st.image(img, use_column_width=True)
                    test_bytes = pil_to_bytes(img)
            else:
                # ✅ Camera with better guidance
                st.info("📷 Allow camera permission when prompted. Then click 'Take a photo' below.")
                cam = st.camera_input("Take a photo", key="cam")
                if cam:
                    try:
                        img = Image.open(cam).convert("RGB")
                        st.image(img, use_column_width=True)
                        test_bytes = pil_to_bytes(img)
                        st.success("✅ Photo captured! Predicting...")
                    except Exception as e:
                        st.error(f"❌ Camera error. Please allow camera permissions in browser settings.")
                else:
                    st.caption("👆 Click the camera button above to capture a photo")
        
        with col2:
            st.markdown("#### 🔮 Prediction Result")
            if test_bytes:
                with st.spinner("🔍 Analyzing image..."):
                    result = api_predict(test_bytes)
                
                if result:
                    top = result["prediction"]
                    conf = result["confidence"]
                    
                    if conf >= 90:
                        border_color = "rgba(34, 197, 94, 0.6)"
                        text_color = "#4ade80"
                    elif conf >= 70:
                        border_color = "rgba(234, 179, 8, 0.6)"
                        text_color = "#fbbf24"
                    else:
                        border_color = "rgba(239, 68, 68, 0.6)"
                        text_color = "#f87171"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(102,126,234,0.15), rgba(118,75,162,0.15)); 
                                border: 2px solid {border_color}; border-radius: 20px; 
                                padding: 2rem; text-align: center; margin-bottom: 1rem;">
                        <p style="color: rgba(255,255,255,0.5); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 2px; margin: 0;">Prediction Result</p>
                        <h1 style="font-size: 3.5rem; color: white; margin: 0.5rem 0; font-weight: 900;">{top}</h1>
                        <p style="font-size: 1.8rem; color: {text_color}; font-weight: 700; margin: 0;">{conf}%</p>
                        <p style="color: rgba(255,255,255,0.4); font-size: 0.8rem; margin-top: 0.5rem;">confidence score</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("##### 📊 Class Probabilities")
                    for prob in result["probabilities"]:
                        pct = round(prob["probability"] * 100, 1)
                        is_top = prob["class"] == top
                        bar_color = "linear-gradient(90deg, #667eea, #764ba2)" if is_top else "linear-gradient(90deg, #4b5563, #6b7280)"
                        
                        st.markdown(f"""
                        <div style="margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="color: white; font-weight: {'700' if is_top else '400'}; font-size: {'1rem' if is_top else '0.85rem'};">
                                    {'🏆 ' if is_top else ''}{prob['class']}
                                </span>
                                <span style="color: {'#4ade80' if is_top else 'rgba(255,255,255,0.6)'}; font-size: 0.9rem; font-weight: {'700' if is_top else '400'};">
                                    {pct}%
                                </span>
                            </div>
                            <div style="background: rgba(255,255,255,0.08); height: {'10px' if is_top else '6px'}; border-radius: {'5px' if is_top else '3px'}; overflow: hidden;">
                                <div style="background: {bar_color}; height: 100%; width: {pct}%; border-radius: {'5px' if is_top else '3px'}; transition: width 0.5s ease;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                else:
                    st.error("❌ Prediction failed. Please check backend connection.")
            else:
                if mode == "📷 Camera":
                    st.info("📷 Click the 'Take a photo' button to capture and get prediction.")
                else:
                    st.info("📁 Upload an image to get prediction.")
    else:
        st.info("👈 Please select and load a model first to start predicting.")


# ═══════════════════════════════════════════════════════════════════════════
# 📊 ANALYTICS PAGE
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.active_tab == "📊 Analytics":
    st.markdown("# 📊 Analytics")
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    if not st.session_state.trained:
        st.info("👈 No model loaded! Train a model or load a saved model first.")
    else:
        if st.session_state.training_history:
            st.markdown("### 📉 Training Progress")
            df = pd.DataFrame(st.session_state.training_history)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.line(df, x='epoch', y='loss', title='Loss Over Time')
                fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                if 'accuracy' in df.columns:
                    fig = px.line(df, x='epoch', y='accuracy', title='Accuracy Over Time')
                    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📉 No training history available for this model.")
        
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        
        cm = st.session_state.confusion_matrix
        classes = st.session_state.class_names
        if cm and classes:
            st.markdown("### 🎯 Confusion Matrix")
            fig = go.Figure(data=go.Heatmap(
                z=cm, x=classes, y=classes,
                colorscale='Purples', text=cm, texttemplate="%{text}",
                textfont={"size": 16, "color": "white"}
            ))
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🎯 No confusion matrix available for this model.")
        
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        
        st.markdown("### 📈 Model Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{st.session_state.accuracy}%")
        with col2:
            st.metric("Number of Classes", len(classes) if classes else 0)
        with col3:
            st.metric("Model Name", st.session_state.selected_model_name or "Current Model")
        
        if classes:
            st.markdown("#### Classes")
            st.write(", ".join(classes))


# ═══════════════════════════════════════════════════════════════════════════
# 💾 MODELS PAGE
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.active_tab == "💾 Models":
    st.markdown("# 💾 Saved Models")
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    result = api("get", "/model/list")
    models = result.get("models", []) if result else []
    
    if not models:
        st.info("No saved models yet. Train and save a model from the **🎓 Train** tab.")
    else:
        st.markdown(f"**{len(models)} model{'s' if len(models) != 1 else ''} saved**")
        
        for m in models:
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                st.markdown(f"**{m['name']}**")
                st.caption(f"Classes: {', '.join(m['classes'])}")
                st.caption(f"Saved: {m.get('saved_at', 'Unknown')}")
            with col2:
                st.metric("Accuracy", f"{round(m['accuracy']*100,1)}%")
            with col3:
                if st.button("📂 Load", key=f"load_{m['name']}"):
                    r = api("post", f"/model/load/{m['name']}")
                    if r:
                        st.session_state.trained = True
                        st.session_state.accuracy = r.get("accuracy", 0)
                        st.session_state.class_names = r.get("classes", [])
                        st.session_state.classes_confirmed = True
                        st.session_state.img_counts = {c: 0 for c in r.get("classes", [])}
                        st.session_state.selected_model_name = m['name']
                        st.session_state.training_history = r.get("history", [])
                        st.session_state.confusion_matrix = r.get("confusion_matrix", [])
                        st.success(f"✅ Model **{m['name']}** loaded!")
                        st.rerun()
            with col4:
                if st.button("🗑️ Delete", key=f"del_{m['name']}"):
                    api("delete", f"/model/{m['name']}")
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# ⚙️ SETTINGS PAGE
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.active_tab == "⚙️ Settings":
    st.markdown("# ⚙️ Settings")
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    current = api("get", "/status")
    cur_settings = current.get("model_settings", {}) if current else {}
    
    st.markdown("### 🎛️ Model Configuration")
    
    with st.form("settings_form"):
        augment = st.checkbox("Data Augmentation", value=cur_settings.get("augment", True),
                              help="Auto-generates flipped, rotated, brightness versions")
        epochs = st.slider("Training Epochs", 10, 100, cur_settings.get("epochs", 50))
        batch_size = st.selectbox("Batch Size", [8, 16, 32], index=1)
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.001
        )
        
        if st.form_submit_button("💾 Save Settings", use_container_width=True):
            r = api("post", "/settings", json={
                "augment": augment,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate
            })
            if r:
                st.success("✅ Settings saved! They will be used on next training.")
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    st.markdown("### ℹ️ About")
    st.markdown("""
    <div class="info-box">
        <h4 style="color: #667eea;">🧠 VisionForge v4.0</h4>
        <ul style="color: rgba(255,255,255,0.7);">
            <li><strong>Backend:</strong> FastAPI + PyTorch + MobileNetV3</li>
            <li><strong>Frontend:</strong> Streamlit + Plotly</li>
            <li><strong>Features:</strong> Transfer Learning, Data Augmentation</li>
            <li><strong>Privacy:</strong> 100% Local Processing</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
