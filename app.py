"""
Nexus Vision — YOLO11 Object Detection Web App
----------------------------------------------
Real-time object detection with YOLO11 (pretrained on COCO, 80 classes).
"""

import time
import warnings
from io import BytesIO

warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# =============================================================================
# Page configuration
# =============================================================================
st.set_page_config(
    page_title="Nexus Vision — AI Object Detection",
    page_icon=":material/visibility:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/ultralytics/ultralytics",
        "Report a bug": "https://github.com/ultralytics/ultralytics/issues",
        "About": "Nexus Vision — YOLO11 object detection built with Streamlit.",
    },
)

# A sane ceiling so a huge phone photo doesn't crawl through the pipeline.
MAX_IMAGE_SIDE = 1920

# =============================================================================
# Custom CSS — modern dark theme, icons, hover effects
# =============================================================================
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0" rel="stylesheet">

    <style>
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp {
            background: radial-gradient(circle at 20% 0%, #1a1f3a 0%, #0d1020 45%, #05060d 100%);
            color: #e6e8f0;
        }

        .material-symbols-rounded {
            font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
            vertical-align: middle;
            font-size: 1.15rem;
            color: #7c8cff;
        }

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
        .hero-icon .material-symbols-rounded { font-size: 2rem; color: #ffffff; }
        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            margin: 0;
            background: linear-gradient(90deg, #ffffff, #b8c0ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-sub { font-size: 0.95rem; color: #9aa4c8; margin-top: 4px; }

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
        .stat-icon .material-symbols-rounded { font-size: 1.5rem; color: #b8c0ff; }
        .stat-value { font-size: 1.6rem; font-weight: 700; color: #ffffff; line-height: 1; }
        .stat-label {
            font-size: 0.8rem;
            color: #9aa4c8;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-top: 4px;
        }

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
        .det-name { font-weight: 600; color: #e6e8f0; flex: 1; text-transform: capitalize; }
        .det-count {
            font-size: 0.85rem;
            color: #9aa4c8;
            padding: 3px 10px;
            border-radius: 8px;
            background: rgba(124,140,255,0.12);
        }
        .det-conf { font-weight: 700; color: #7cffb2; font-variant-numeric: tabular-nums; }
        .det-range { font-size: 0.78rem; color: #9aa4c8; margin-left: 6px; }

        .section-title {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1.15rem;
            font-weight: 700;
            color: #e6e8f0;
            margin: 24px 0 12px;
        }

        .stButton > button {
            background: linear-gradient(135deg, #7c8cff, #a05aff);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 10px 22px;
            font-weight: 600;
            transition: all 0.25s ease;
        }
        .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(124,140,255,0.4); }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d1020 0%, #141834 100%);
            border-right: 1px solid rgba(124,140,255,0.15);
        }
        .stRadio > div { gap: 12px; }
        .stRadio label { transition: all 0.2s ease; }
        .stRadio label:hover { color: #b8c0ff; }
        .stFileUploader, .stCameraInput { border-radius: 14px; transition: all 0.25s ease; }
        .stFileUploader:hover, .stCameraInput:hover { box-shadow: 0 0 24px rgba(124,140,255,0.15); }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Helper functions
# =============================================================================
def icon(name: str, size: str = "1.15rem") -> str:
    """Return an HTML span for a Material Symbols icon, styled consistently."""
    return (
        f'<span class="material-symbols-rounded" '
        f'style="font-size:{size}; font-variation-settings: '
        f"'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;\">{name}</span>"
    )


@st.cache_resource(show_spinner=False)
def load_model(weights: str) -> YOLO:
    return YOLO(weights)


def resize_if_needed(image: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> Image.Image:
    """Downscale oversized uploads so inference stays fast and memory-safe."""
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    image = image.copy()
    image.thumbnail((max_side, max_side))
    return image


def run_detection(model: YOLO, image: Image.Image, conf: float, iou: float, max_det: int):
    start = time.time()
    results = model(np.array(image), conf=conf, iou=iou, max_det=max_det)
    elapsed = time.time() - start

    result = results[0]
    annotated_bgr = result.plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            name = model.names.get(cls_id, f"class_{cls_id}")
            detections.append(
                (
                    name,
                    float(box.conf[0]),
                    box.xyxy[0].cpu().numpy().astype(int).tolist(),
                )
            )
    return annotated_rgb, detections, elapsed


def summarize(detections):
    summary = {}
    for name, score, _ in detections:
        summary.setdefault(name, []).append(score)
    return {
        n: {
            "count": len(s),
            "avg_conf": sum(s) / len(s),
            "max_conf": max(s),
            "min_conf": min(s),
        }
        for n, s in summary.items()
    }


# =============================================================================
# Sidebar
# =============================================================================
MODEL_OPTIONS = {
    "Nano — fastest, least accurate (yolov8n.pt)": "yolov8n.pt",
    "Small — balanced (yolov8s.pt)": "yolov8s.pt",
    "Medium — recommended, best balance (yolo11m.pt)": "yolo11m.pt",
    "Large — high accuracy, slower (yolo11l.pt)": "yolo11l.pt",
}

with st.sidebar:
    st.markdown(
        f'<div class="section-title">{icon("tune", "1.3rem")} Settings</div>',
        unsafe_allow_html=True,
    )

    model_label = st.selectbox(
        "Model size",
        list(MODEL_OPTIONS.keys()),
        index=2,  # Default to Medium (yolo11m.pt)
        help="YOLO11m is the recommended default: best accuracy/speed balance.",
    )
    weights_file = MODEL_OPTIONS[model_label]

    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.05,
        help="Only detections above this score are shown.",
    )

    iou_threshold = st.slider(
        "IoU threshold (NMS)",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Controls how aggressively overlapping boxes for the same object are merged.",
    )

    max_det = st.slider(
        "Max detections",
        min_value=10,
        max_value=300,
        value=100,
        step=10,
        help="Upper limit on how many objects can be reported for one image.",
    )

    with st.spinner("Loading model..."):
        try:
            model = load_model(weights_file)
            model_loaded = True
        except Exception as e:
            model_loaded = False
            st.error(f"Couldn't load `{weights_file}`: {e}")

    COCO_CLASSES = list(model.names.values()) if model_loaded else []

    exclude_classes = st.multiselect(
        "Exclude classes",
        COCO_CLASSES,
        help="Detections of these classes will be filtered out of the results.",
    )

    with st.expander("COCO class library"):
        st.caption(", ".join(COCO_CLASSES) if COCO_CLASSES else "Model not loaded.")

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
    st.caption("YOLO11 · Streamlit · COCO (80 classes)")

if not model_loaded:
    st.stop()

# =============================================================================
# Hero header
# =============================================================================
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-icon">{icon("visibility", "2rem")}</div>
        <div>
            <div class="hero-title">Nexus Vision</div>
            <div class="hero-sub">AI-powered object detection · YOLO11 · 80 COCO classes</div>
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

if image is not None:
    original_size = image.size
    image = resize_if_needed(image)
    if image.size != original_size:
        st.caption(
            f"Image resized from {original_size[0]}×{original_size[1]} to "
            f"{image.size[0]}×{image.size[1]} for faster processing."
        )

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

    with st.spinner("Running YOLO11 inference..."):
        try:
            annotated_rgb, detections, inference_time = run_detection(
                model, image, conf_threshold, iou_threshold, max_det
            )
        except Exception as e:
            st.error(f"Detection failed: {e}")
            st.stop()

    if exclude_classes:
        detections = [d for d in detections if d[0] not in exclude_classes]

    with col_detected:
        st.markdown(
            f'<div class="section-title">{icon("target", "1.2rem")} Detected</div>',
            unsafe_allow_html=True,
        )
        st.image(annotated_rgb, use_container_width=True)
        st.caption(f"⏱ Inference time: {inference_time:.2f}s")

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
                <div class="stat">
                    <div class="stat-icon">{icon("timer", "1.5rem")}</div>
                    <div>
                        <div class="stat-value">{inference_time:.2f}s</div>
                        <div class="stat-label">Inference time</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="section-title">{icon("list_alt", "1.2rem")} Detection breakdown</div>',
            unsafe_allow_html=True,
        )

        sorted_summary = sorted(summary.items(), key=lambda x: -x[1]["count"])
        rows_html = "".join(
            f"""
            <div class="det-item">
                {icon("label")}
                <span class="det-name">{name}</span>
                <span class="det-count">× {info['count']}</span>
                <span class="det-conf">{info['avg_conf']:.1%}</span>
                <span class="det-range">(min {info['min_conf']:.0%} · max {info['max_conf']:.0%})</span>
            </div>
            """
            for name, info in sorted_summary
        )
        st.markdown(rows_html, unsafe_allow_html=True)

        with st.expander("Bounding box coordinates"):
            for name, score, (x1, y1, x2, y2) in detections:
                st.markdown(
                    f'{icon("crop_free", "1rem")} **{name}** ({score:.1%}) — `[{x1}, {y1}, {x2}, {y2}]`',
                    unsafe_allow_html=True,
                )

        annotated_pil = Image.fromarray(annotated_rgb)
        buf = BytesIO()
        annotated_pil.save(buf, format="PNG")
        st.download_button(
            "⬇️ Download annotated image",
            data=buf.getvalue(),
            file_name="nexus_vision_result.png",
            mime="image/png",
        )
    else:
        st.warning(
            "No objects detected. Try lowering the confidence threshold in the sidebar."
        )
        st.caption(
            "Tip: YOLO11 tends to struggle with drawings, cartoons, heavy motion blur, "
            "and very low-light images."
        )
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