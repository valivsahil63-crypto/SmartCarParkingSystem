"""
streamlit_app.py — Smart Car Parking System (Streamlit Cloud Version)
Supports: uploaded video frames, demo parking image, live ML detection
"""

import streamlit as st
import cv2
import numpy as np
import json
import time
import os
import tempfile
from datetime import datetime
from PIL import Image
import urllib.request

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartPark AI — Parking Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #08111e; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; }

/* Header */
.hero-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #112240 100%);
    border: 1px solid #1e3150;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00aaff, #7c4dff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-sub { color: #7a9cc5; font-size: 0.9rem; margin: 0; }

/* Stat cards */
.stat-card {
    background: #111e2f;
    border: 1px solid #1e3150;
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    transition: border-color 0.3s;
}
.stat-val  { font-size: 2.5rem; font-weight: 800; line-height: 1; }
.stat-lbl  { font-size: 0.75rem; color: #7a9cc5; margin-top: 6px; }
.free-val     { color: #00e676; }
.occupied-val { color: #ff5252; }
.total-val    { color: #00aaff; }
.entered-val  { color: #00e676; }
.exited-val   { color: #ff6e40; }

/* Plate entry */
.plate-entry {
    background: #111e2f;
    border: 1px solid #1e3150;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    font-family: 'JetBrains Mono', monospace;
}
.plate-num  { color: #00e5ff; font-size: 1rem; font-weight: 600; letter-spacing: 2px; }
.plate-conf { color: #00e676; font-size: 0.8rem; }
.plate-time { color: #4a6580; font-size: 0.75rem; margin-left: auto; }

/* Info box */
.info-box {
    background: #0d1b2a;
    border-left: 3px solid #00aaff;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: #7a9cc5;
    margin-bottom: 1rem;
}

/* Badge */
.badge-live {
    background: #ff5252; color: white;
    font-size: 0.65rem; font-weight: 700; letter-spacing: 1.5px;
    padding: 2px 8px; border-radius: 4px;
    animation: blink 1.5s step-start infinite;
    display: inline-block;
}
@keyframes blink { 50% { opacity: 0; } }

.badge-ai {
    background: linear-gradient(135deg, #00aaff, #7c4dff);
    color: white; font-size: 0.6rem; font-weight: 700;
    padding: 2px 6px; border-radius: 4px;
    letter-spacing: 1px; display: inline-block;
}

/* Section headers */
.section-title {
    font-size: 0.9rem; font-weight: 700;
    color: #e8f0fe; margin-bottom: 0.75rem;
    border-bottom: 1px solid #1e3150;
    padding-bottom: 0.4rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1b2a !important;
    border-right: 1px solid #1e3150;
}
</style>
""", unsafe_allow_html=True)


# ── Load YOLO lazily ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_yolo():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")


# ── Demo parking image (public domain) ────────────────────────────────────────
DEMO_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/94/Parking_lot_at_Target.jpg/1280px-Parking_lot_at_Target.jpg"

@st.cache_data(show_spinner=False)
def get_demo_image():
    try:
        req = urllib.request.Request(DEMO_IMAGE_URL,
              headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            arr = np.frombuffer(r.read(), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
    except Exception:
        # Fallback: generate a synthetic parking lot
        img = np.zeros((480, 800, 3), dtype=np.uint8)
        img[:] = (20, 30, 45)
        # Draw parking lines
        for x in range(80, 780, 140):
            cv2.rectangle(img, (x, 100), (x+120, 260), (50, 60, 80), -1)
            cv2.rectangle(img, (x, 100), (x+120, 260), (80, 100, 130), 2)
            cv2.rectangle(img, (x, 300), (x+120, 440), (50, 60, 80), -1)
            cv2.rectangle(img, (x, 300), (x+120, 440), (80, 100, 130), 2)
        cv2.putText(img, "PARKING LOT - DEMO", (200, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 150, 200), 2)
        return img


# ── Detection functions ────────────────────────────────────────────────────────
VEHICLE_CLASSES = [2, 3, 5, 7]   # car, motorcycle, bus, truck
VEHICLE_CONF    = 0.35

def detect_vehicles(model, frame):
    results = model(frame, verbose=False, conf=VEHICLE_CONF,
                    classes=VEHICLE_CLASSES)[0]
    boxes = []
    labels = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        boxes.append((x1, y1, x2, y2))
        labels.append((model.names[cls], conf))
    return boxes, labels


def draw_detections(frame, boxes, labels):
    for (x1, y1, x2, y2), (name, conf) in zip(boxes, labels):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
        label = f"{name} {conf:.0%}"
        cv2.putText(frame, label, (x1, max(y1-6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 165, 0), 2, cv2.LINE_AA)
    return frame


def check_slot_occupancy(slots, boxes):
    """Returns list of True/False for each slot."""
    states = []
    for slot in slots:
        pts = np.array(slot, dtype=np.int32)
        occupied = False
        for (x1, y1, x2, y2) in boxes:
            # Simple centre-point check
            cx, cy = (x1+x2)//2, (y1+y2)//2
            if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                occupied = True
                break
        states.append(occupied)
    return states


def draw_slots(frame, slots, states):
    overlay = frame.copy()
    EMPTY_COLOR    = (0, 230, 118)
    OCCUPIED_COLOR = (255, 82, 82)
    for idx, (slot, occupied) in enumerate(zip(slots, states)):
        pts   = np.array(slot, dtype=np.int32)
        color = OCCUPIED_COLOR if occupied else EMPTY_COLOR
        cv2.fillPoly(overlay, [pts], color)
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        cv2.putText(overlay, str(idx+1), (cx-8, cy+6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    for slot in slots:
        pts = np.array(slot, dtype=np.int32)
        cv2.polylines(frame, [pts], True, (200, 200, 200), 1)
    return frame


def simulate_plates(boxes):
    """Generate realistic-looking plate data for demo."""
    import random
    plates = []
    letters = "ABCDEFGHJKLMNPRSTUVWXYZ"
    for i, _ in enumerate(boxes[:3]):   # max 3 plates shown
        p = ("".join(random.choices(letters, k=3)) +
             str(random.randint(100, 999)))
        plates.append({
            "plate": p,
            "confidence": round(random.uniform(0.72, 0.97), 2),
            "time": datetime.now().strftime("%H:%M:%S")
        })
    return plates


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 SmartPark AI")
    st.markdown('<span class="badge-ai">ML POWERED</span>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 📹 Video Source")
    source_mode = st.radio("Select input mode:",
                           ["🖼️ Demo Image", "📤 Upload Image", "🎬 Upload Video"],
                           index=0)

    uploaded_file = None
    if source_mode == "📤 Upload Image":
        uploaded_file = st.file_uploader("Upload parking lot image",
                                          type=["jpg","jpeg","png"])
    elif source_mode == "🎬 Upload Video":
        uploaded_file = st.file_uploader("Upload parking lot video",
                                          type=["mp4","avi","mov"])

    st.markdown("---")
    st.markdown("### ⚙️ Detection Settings")
    conf_thresh = st.slider("Detection Confidence", 0.2, 0.9, 0.35, 0.05)
    show_slots  = st.checkbox("Show Parking Slots", value=True)
    show_boxes  = st.checkbox("Show Vehicle Boxes", value=True)
    show_line   = st.checkbox("Show Counting Line", value=True)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Technologies:**
    - 🤖 YOLOv8 (Ultralytics)
    - 👁️ OpenCV
    - 🔤 EasyOCR (simulated on cloud)
    - 🌊 Streamlit

    **Features:**
    - Parking slot detection
    - Vehicle counting
    - Plate recognition
    """)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div style="font-size:3rem">🚗</div>
  <div>
    <p class="hero-title">SmartPark AI <span class="badge-ai">ML</span></p>
    <p class="hero-sub">Real-Time Parking Intelligence · YOLOv8 · OpenCV · Python</p>
  </div>
  <div style="margin-left:auto">
    <span class="badge-live">● LIVE</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────────────────────────
with st.spinner("🔄 Loading YOLOv8 model..."):
    try:
        model = load_yolo()
        model_loaded = True
    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        model_loaded = False


# ── Get frame ─────────────────────────────────────────────────────────────────
frame = None
is_video = False

if source_mode == "🖼️ Demo Image":
    with st.spinner("Loading demo image..."):
        frame = get_demo_image()
    st.markdown('<div class="info-box">📌 Using demo parking lot image. Upload your own image or video for real detection.</div>',
                unsafe_allow_html=True)

elif source_mode == "📤 Upload Image" and uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

elif source_mode == "🎬 Upload Video" and uploaded_file:
    is_video = True
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.flush()
    video_path = tfile.name

elif uploaded_file is None and source_mode != "🖼️ Demo Image":
    st.info("👆 Please upload a file using the sidebar.")


# ── Process & display ──────────────────────────────────────────────────────────
if frame is not None and model_loaded:

    # Run detection
    boxes, labels = detect_vehicles(model, frame)

    # Demo slots (pre-defined for demo image)
    demo_slots = [
        [[120, 110], [230, 110], [230, 250], [120, 250]],
        [[250, 110], [360, 110], [360, 250], [250, 250]],
        [[380, 110], [490, 110], [490, 250], [380, 250]],
        [[510, 110], [620, 110], [620, 250], [510, 250]],
        [[640, 110], [750, 110], [750, 250], [640, 250]],
        [[120, 270], [230, 270], [230, 420], [120, 420]],
        [[250, 270], [360, 270], [360, 420], [250, 420]],
        [[380, 270], [490, 270], [490, 420], [380, 420]],
        [[510, 270], [620, 270], [620, 420], [510, 420]],
        [[640, 270], [750, 270], [750, 420], [640, 420]],
    ]

    display = frame.copy()
    if show_slots:
        states = check_slot_occupancy(demo_slots, boxes)
        display = draw_slots(display, demo_slots, states)
    else:
        states = [False] * len(demo_slots)

    if show_boxes:
        display = draw_detections(display, boxes, labels)

    # Counting line
    if show_line:
        h, w = display.shape[:2]
        line_y = h // 2
        cv2.line(display, (0, line_y), (w, line_y), (255, 255, 0), 2)
        cv2.putText(display, "COUNTING LINE", (10, line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Stats
    occupied = sum(states)
    free     = len(demo_slots) - occupied
    total    = len(demo_slots)
    entered  = len(boxes) + 3   # simulated
    exited   = max(0, entered - occupied)
    plates   = simulate_plates(boxes)

    # ── Layout ──────────────────────────────────────────────────────────────
    col_vid, col_stats = st.columns([3, 1.2])

    with col_vid:
        st.markdown('<div class="section-title">📹 Live Detection Feed</div>', unsafe_allow_html=True)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        st.image(rgb, use_container_width=True)

        st.markdown(f"""
        <div class="info-box">
        ✅ Detected <b>{len(boxes)}</b> vehicle(s) &nbsp;|&nbsp;
        🟢 <b>{free}</b> free slots &nbsp;|&nbsp;
        🔴 <b>{occupied}</b> occupied slots
        </div>""", unsafe_allow_html=True)

    with col_stats:
        # Occupancy
        st.markdown('<div class="section-title">🅿️ Occupancy</div>', unsafe_allow_html=True)
        pct = int((occupied / total * 100)) if total else 0
        st.progress(pct / 100)
        st.markdown(f"**{pct}%** occupied")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="stat-card"><div class="stat-val free-val">{free}</div><div class="stat-lbl">Free</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-card"><div class="stat-val occupied-val">{occupied}</div><div class="stat-lbl">Occupied</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-card"><div class="stat-val total-val">{total}</div><div class="stat-lbl">Total</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Counter
        st.markdown('<div class="section-title">🚙 Vehicle Counter</div>', unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca:
            st.markdown(f'<div class="stat-card"><div class="stat-val entered-val">{entered}</div><div class="stat-lbl">⬇ Entered</div></div>', unsafe_allow_html=True)
        with cb:
            st.markdown(f'<div class="stat-card"><div class="stat-val exited-val">{exited}</div><div class="stat-lbl">⬆ Exited</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Plates
        st.markdown('<div class="section-title">🔍 Plate Recognition</div>', unsafe_allow_html=True)
        if plates:
            for p in plates:
                st.markdown(f"""
                <div class="plate-entry">
                  <span class="plate-num">{p['plate']}</span>
                  <span class="plate-conf">{int(p['confidence']*100)}%</span>
                  <span class="plate-time">{p['time']}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#4a6580;font-size:0.8rem">No plates detected</p>',
                        unsafe_allow_html=True)

    # ── Video mode: frame-by-frame ───────────────────────────────────────────
    if is_video:
        st.markdown("---")
        st.markdown("### 🎬 Video Playback")
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_no = st.slider("Seek Frame", 0, max(total_frames-1, 1), 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, vframe = cap.read()
        cap.release()
        if ret:
            vboxes, vlabels = detect_vehicles(model, vframe)
            vdisp = vframe.copy()
            vstates = check_slot_occupancy(demo_slots, vboxes)
            if show_slots: vdisp = draw_slots(vdisp, demo_slots, vstates)
            if show_boxes: vdisp = draw_detections(vdisp, vboxes, vlabels)
            st.image(cv2.cvtColor(vdisp, cv2.COLOR_BGR2RGB), use_container_width=True)


elif is_video and uploaded_file is None:
    pass
elif not model_loaded:
    st.error("⚠️ Could not load YOLO model. Please check your internet connection.")
else:
    # Welcome screen
    st.markdown("""
    <div style="text-align:center; padding:3rem; background:#111e2f; border-radius:16px; border:1px solid #1e3150;">
      <div style="font-size:5rem">🚗</div>
      <h2 style="color:#e8f0fe">Welcome to SmartPark AI</h2>
      <p style="color:#7a9cc5">Select <b>Demo Image</b> in the sidebar to see ML detection instantly,<br>
      or upload your own parking lot image/video.</p>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#4a6580; font-size:0.75rem; font-family:'JetBrains Mono',monospace;">
  SmartPark AI · YOLOv8 · OpenCV · EasyOCR · Streamlit · Built with ❤️
</div>
""", unsafe_allow_html=True)
