"""
Nexus Vision — YOLOv8 Object Detection Web App
----------------------------------------------
Real-time object detection with YOLOv8 (pretrained on COCO, 80 classes).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


# =============================================================================
# Page configuration
# =============================================================================
st.set_page_config(
    page_title="Nexus Vision — AI Object Detection",
    page_icon=":material/visibility:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Custom CSS — modern dark theme, icons, hover effects
# =============================================================================
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0" rel="stylesheet">

    <style>
        /* ---------- Global ---------- */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        .stApp {
            background: radial-gradient(circle at 20% 0%, #1a1f3a 0%, #0d1020 45%, #05060d 100%);
            color: #e6e8f0;
        }

        /* ---------- Icons ---------- */
        .material-symbols-rounded {
            font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
            vertical-align: middle;
            font-size: 1.15rem;
            color: #7c8cff;
        }

        /* ---------- Hero header ---------- */
        .hero {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 28px 32px;
            border-radius: 20px;
            background: linear-gradient(135deg, rgba(124,140,255,0.15), rgba(160,90,255,0.08));
            border: 1px solid rgba(124,140,255,0.25);
            backdrop-filter: blur(12px);
            margin-bottom: 24px;
            transition: all 0.3s ease;
        }
        .hero:hover {
            border-color: rgba(124,140,255,0.55);
            box-shadow: 0 0 30px rgba(124,140,255,0.18);
        }
        .hero-icon {
            width: 56px;
            height: 56px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 16px;
            background: linear-gradient(135deg, #7c8cff, #a05aff);
            box-shadow: 0 0 20px rgba(124,140,255,0.4);
        }
        .hero-icon .material-symbols-rounded {
            font-size: 2rem;
            color: #ffffff;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            margin: 0;
            background: linear-gradient(90deg, #ffffff, #b8c0ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-sub {
            font-size: 0.95rem;
            color: #9aa4c8;
            margin-top: 4px;
        }

        /* ---------- Card ---------- */
        .card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(124,140,255,0.18);
            border-radius: 16px;
            padding: 20px;
            transition: all 0.25s ease;
            margin-bottom: 14px;
        }
        .card:hover {
            border-color: rgba(124,140,255,0.55);
            background: rgba(124,140,255,0.06);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.35);
        }

        /* ---------- Stat cards ---------- */
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
            margin: 18px 0;
        }
        .stat {
            background: linear-gradient(135deg, rgba(124,140,255,0.10), rgba(160,90,255,0.05));
            border: 1px solid rgba(124,140,255,0.22);
            border-radius: 14px;
            padding: 18px 20px;
            display: flex;
            align-items: center;
            gap: 14px;
            transition: all 0.25s ease;
        }
        .stat:hover {
            border-color: rgba(124,140,255,0.6);
            box-shadow: 0 0 22px rgba(124,140,255,0.18);
            transform: translateY(-3px);
        }
        .stat-icon {
            width: 44px;
            height: 44px;
            border-radius: 12px;
            background: rgba(124,140,255,0.18);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .stat-icon .material-symbols-rounded {
            font-size: 1.5rem;
            color: #b8c0ff;
        }
        .stat-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #ffffff;
            line-height: 1;
        }
        .stat-label {
            font-size: 0.8rem;
            color: #9aa4c8;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-top: 4px;
        }

        /* ---------- Detection list ---------- */
        .det-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px 16px;
            border-radius: 12px;
            background: rgba(255,255,255,0.03);
            border-left: 3px solid #7c8cff;
            margin-bottom: 8px;
            transition: all 0.2s ease;
        }
        .det-item:hover {
            background: rgba(124,140,255,0.08);
            border-left-color: #a05aff;
            transform: translateX(4px);
        }
        .det-name {
            font-weight: 600;
            color: #e6e8f0;
            flex: 1;
            text-transform: capitalize;
        }
        .det-count {
            font-size: 0.85rem;
            color: #9aa4c8;
            padding: 3px 10px;
            border-radius: 8px;
            background: rgba(124,140,255,0.12);
        }
        .det-conf {
            font-weight: 700;
            color: #7cffb2;
            font-variant-numeric: tabular-nums;
        }

        /* ---------- Section titles ---------- */
        .section-title {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1.15rem;
            font-weight: 700;
            color: #e6e8f0;
            margin: 24px 0 12px;
        }

        /* ---------- Buttons ---------- */
        .stButton > button {
            background: linear-gradient(135deg, #7c8cff, #a05aff);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 10px 22px;
            font-weight: 600;
            transition: all 0.25s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(124,140,255,0.4);
        }

        /* ---------- Streamlit widget tweaks ---------- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d1020 0%, #141834 100%);
            border-right: 1px solid rgba(124,140,255,0.15);
        }
        .stRadio > div { gap: 12px; }
        .stRadio label { transition: all 0.2s ease; }
        .stRadio label:hover { color: #b8c0ff; }
        .stFileUploader, .stCameraInput {
            border-radius: 14px;
            transition: all 0.25s ease;
        }
        .stFileUploader:hover, .stCameraInput:hover {
            box-shadow: 0 0 24px rgba(124,140,255,0.15);
        }

        /* Hide Streamlit chrome */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Model loading
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_model() -> YOLO:
    return YOLO("yolov8n.pt")


model = load_model()
COCO_CLASSES = list(model.names.values())


# =============================================================================
# Helper functions
# =============================================================================
def run_detection(image: Image.Image, conf: float):
    results = model(np.array(image), conf=conf, max_det=50)
    result = results[0]
    annotated_bgr = result.plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            detections.append(
                (
                    model.names[cls_id],
                    float(box.conf[0]),
                    box.xyxy[0].cpu().numpy().astype(int).tolist(),
                )
            )
    return annotated_rgb, detections


def summarize(detections):
    summary = {}
    for name, score, _ in detections:
        summary.setdefault(name, []).append(score)
    return {
        n: {"count": len(s), "avg_conf": sum(s) / len(s)}
        for n, s in summary.items()
    }


def icon(name: str, size: str = "1.15rem") -> str:
    """Return an HTML span for a Material Symbols icon."""
    return (
        f'<span class="material-symbols-rounded" '
        f'style="font-size:{size}">{name}</span>'
    )


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown(
        f'<div class="section-title">{icon("tune", "1.3rem")} Settings</div>',
        unsafe_allow_html=True,
    )

    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.05,
        help="Only detections above this score are shown.",
    )

    with st.expander("COCO class library"):
        st.caption(", ".join(COCO_CLASSES))

    st.markdown("---")
    st.markdown(
        f'<div class="section-title">{icon("info", "1.3rem")} About confidence</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "The percentage next to each detection is a **confidence score**, "
        "not a calibrated probability. It reflects the model's certainty "
        "about that detection (objectness × class probability)."
    )

    st.markdown("---")
    st.caption("YOLOv8 · Streamlit · COCO (80 classes)")


# =============================================================================
# Hero header
# =============================================================================
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-icon">{icon("visibility", "2rem")}</div>
        <div>
            <div class="hero-title">Nexus Vision</div>
            <div class="hero-sub">AI-powered object detection · YOLOv8 · 80 COCO classes</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="section-title">{icon("add_photo_alternate", "1.3rem")} Provide an image</div>',
    unsafe_allow_html=True,
)

input_mode = st.radio(
    "Input method",
    ("Upload image", "Capture from camera"),
    horizontal=True,
    label_visibility="collapsed",
)

image = None

if input_mode == "Upload image":
    uploaded_file = st.file_uploader(
        "Drop an image here",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Could not read the uploaded file: {e}")

else:
    camera_file = st.camera_input("Take a picture", label_visibility="collapsed")
    if camera_file is not None:
        try:
            image = Image.open(camera_file).convert("RGB")
        except Exception as e:
            st.error(f"Could not read the captured image: {e}")


# =============================================================================
# Detection & results
# =============================================================================
if image is not None:
    col_original, col_detected = st.columns(2, gap="large")

    with col_original:
        st.markdown(
            f'<div class="section-title">{icon("image", "1.2rem")} Original</div>',
            unsafe_allow_html=True,
        )
        st.image(image, use_container_width=True)

    with st.spinner("Running YOLOv8 inference..."):
        try:
            annotated_rgb, detections = run_detection(image, conf_threshold)
        except Exception as e:
            st.error(f"Detection failed: {e}")
            st.stop()

    with col_detected:
        st.markdown(
            f'<div class="section-title">{icon("target", "1.2rem")} Detected</div>',
            unsafe_allow_html=True,
        )
        st.image(annotated_rgb, use_container_width=True)

    # ---------- Stat cards ----------
    if detections:
        summary = summarize(detections)
        total = len(detections)
        n_classes = len(summary)
        avg_conf = sum(d[1] for d in detections) / total

        st.markdown(
            f"""
            <div class="stat-grid">
                <div class="stat">
                    <div class="stat-icon">{icon("category", "1.5rem")}</div>
                    <div>
                        <div class="stat-value">{total}</div>
                        <div class="stat-label">Objects found</div>
                    </div>
                </div>
                <div class="stat">
                    <div class="stat-icon">{icon("dataset", "1.5rem")}</div>
                    <div>
                        <div class="stat-value">{n_classes}</div>
                        <div class="stat-label">Unique classes</div>
                    </div>
                </div>
                <div class="stat">
                    <div class="stat-icon">{icon("verified", "1.5rem")}</div>
                    <div>
                        <div class="stat-value">{avg_conf:.1%}</div>
                        <div class="stat-label">Avg confidence</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------- Detection list ----------
        st.markdown(
            f'<div class="section-title">{icon("list_alt", "1.2rem")} Detection breakdown</div>',
            unsafe_allow_html=True,
        )

        sorted_summary = sorted(summary.items(), key=lambda x: -x[1]["count"])
        rows_html = "".join(
            f"""
            <div class="det-item">
                <span class="material-symbols-rounded">label</span>
                <span class="det-name">{name}</span>
                <span class="det-count">× {info['count']}</span>
                <span class="det-conf">{info['avg_conf']:.1%}</span>
            </div>
            """
            for name, info in sorted_summary
        )
        st.markdown(rows_html, unsafe_allow_html=True)

        # ---------- Raw coordinates ----------
        with st.expander("Bounding box coordinates"):
            for name, score, (x1, y1, x2, y2) in detections:
                st.markdown(
                    f'<span class="material-symbols-rounded">crop_free</span> '
                    f'**{name}** ({score:.1%}) — `[{x1}, {y1}, {x2}, {y2}]`',
                    unsafe_allow_html=True,
                )
    else:
        st.warning("No objects detected. Try lowering the confidence threshold in the sidebar.")

else:
    st.markdown(
        f"""
        <div class="card" style="text-align:center; padding: 40px 20px;">
            {icon("upload_file", "2.5rem")}
            <div style="margin-top: 12px; color: #9aa4c8;">
                Upload an image or use your camera to get started.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )