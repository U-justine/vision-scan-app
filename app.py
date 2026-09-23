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
from PIL import Image, ImageOps
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

MAX_IMAGE_SIDE = 1920

# =============================================================================
# Modern design system (deep slate + electric teal)
# =============================================================================
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0" rel="stylesheet">

    <style>
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp {
            background: linear-gradient(160deg, #0b1220 0%, #0f172a 40%, #0a0f1c 100%);
            color: #e2e8f0;
        }

        .material-symbols-rounded {
            font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
            vertical-align: middle;
            font-size: 1.15rem;
            color: #2dd4bf;
        }

        .hero {
            display: flex;
            align-items: center;
            gap: 20px;
            padding: 32px 36px;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(45,212,191,0.12), rgba(14,165,233,0.08));
            border: 1px solid rgba(45,212,191,0.25);
            backdrop-filter: blur(16px);
            margin-bottom: 28px;
            transition: all 0.3s ease;
        }
        .hero:hover {
            border-color: rgba(45,212,191,0.55);
            box-shadow: 0 0 40px rgba(45,212,191,0.15);
        }
        .hero-icon {
            width: 60px;
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 18px;
            background: linear-gradient(135deg, #2dd4bf, #0ea5e9);
            box-shadow: 0 0 24px rgba(45,212,191,0.45);
        }
        .hero-icon .material-symbols-rounded { font-size: 2.1rem; color: #0f172a; }
        .hero-title {
            font-size: 2.15rem;
            font-weight: 800;
            margin: 0;
            background: linear-gradient(90deg, #f8fafc, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-sub { font-size: 1rem; color: #94a3b8; margin-top: 6px; letter-spacing: 0.02em; }

        .card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(45,212,191,0.15);
            border-radius: 18px;
            padding: 22px;
            transition: all 0.25s ease;
            margin-bottom: 16px;
        }
        .card:hover {
            border-color: rgba(45,212,191,0.4);
            background: rgba(45,212,191,0.05);
            transform: translateY(-2px);
        }

        .stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 16px;
            margin: 20px 0;
        }
        .stat {
            background: linear-gradient(145deg, rgba(45,212,191,0.08), rgba(14,165,233,0.04));
            border: 1px solid rgba(45,212,191,0.2);
            border-radius: 16px;
            padding: 20px;
            display: flex;
            align-items: center;
            gap: 14px;
            transition: all 0.25s ease;
        }
        .stat:hover {
            border-color: rgba(45,212,191,0.5);
            box-shadow: 0 0 28px rgba(45,212,191,0.12);
            transform: translateY(-3px);
        }
        .stat-icon {
            width: 46px;
            height: 46px;
            border-radius: 14px;
            background: rgba(45,212,191,0.15);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .stat-icon .material-symbols-rounded { font-size: 1.5rem; color: #5eead4; }
        .stat-value { font-size: 1.65rem; font-weight: 700; color: #f1f5f9; line-height: 1; }
        .stat-label {
            font-size: 0.78rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-top: 4px;
        }

        .det-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 13px 18px;
            border-radius: 14px;
            background: rgba(255,255,255,0.025);
            border-left: 3px solid #2dd4bf;
            margin-bottom: 9px;
            transition: all 0.2s ease;
        }
        .det-item:hover {
            background: rgba(45,212,191,0.07);
            border-left-color: #0ea5e9;
            transform: translateX(5px);
        }
        .det-name { font-weight: 600; color: #e2e8f0; flex: 1; text-transform: capitalize; }
        .det-count {
            font-size: 0.84rem;
            color: #94a3b8;
            padding: 3px 11px;
            border-radius: 9px;
            background: rgba(45,212,191,0.12);
        }
        .det-conf { font-weight: 700; color: #5eead4; font-variant-numeric: tabular-nums; }
        .det-range { font-size: 0.76rem; color: #64748b; margin-left: 6px; }

        .section-title {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1.18rem;
            font-weight: 700;
            color: #f1f5f9;
            margin: 26px 0 14px;
        }

        .tip-card {
            background: rgba(45,212,191,0.06);
            border: 1px solid rgba(45,212,191,0.18);
            border-radius: 14px;
            padding: 14px 16px;
            margin-bottom: 12px;
            font-size: 0.9rem;
            line-height: 1.45;
            color: #cbd5e1;
        }
        .tip-card strong { color: #5eead4; }

        .stButton > button {
            background: linear-gradient(135deg, #2dd4bf, #0ea5e9);
            color: #0f172a;
            border: none;
            border-radius: 14px;
            padding: 11px 24px;
            font-weight: 600;
            transition: all 0.25s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 28px rgba(45,212,191,0.35);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
            border-right: 1px solid rgba(45,212,191,0.12);
        }

        .copyright {
            text-align: center;
            color: #475569;
            font-size: 0.85rem;
            margin-top: 48px;
            padding: 24px 0 12px;
            border-top: 1px solid rgba(45,212,191,0.1);
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Helpers
# =============================================================================
def icon(name: str, size: str = "1.15rem") -> str:
    return (
        f'<span class="material-symbols-rounded" '
        f'style="font-size:{size}; font-variation-settings: '
        f"'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;\">{name}</span>"
    )


@st.cache_resource(show_spinner=False)
def load_model(weights: str) -> YOLO:
    return YOLO(weights)


def resize_if_needed(image: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    image = image.copy()
    image.thumbnail((max_side, max_side))
    return image


def run_detection(model: YOLO, image: Image.Image, conf: float, iou: float, max_det: int, device: str):
    start = time.time()
    results = model(
        np.array(image),
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
        verbose=False,
    )
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
# Sidebar — Controls + Did you know? + COCO class library
# =============================================================================
with st.sidebar:
    st.markdown(
        f'<div class="section-title">{icon("tune", "1.3rem")} Controls</div>',
        unsafe_allow_html=True,
    )

    model_choice = st.selectbox(
        "Model",
        ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt"],
        index=2,
        help="yolo11m is the recommended balance of speed and accuracy.",
    )

    device = st.selectbox(
        "Device",
        ["cpu", "cuda", "mps"],
        index=0,
        help="Use cuda or mps if you have a compatible GPU.",
    )

    conf_threshold = st.slider("Confidence", 0.10, 0.90, 0.40, 0.05)
    iou_threshold = st.slider("IoU (NMS)", 0.10, 0.90, 0.45, 0.05)
    max_det = st.slider("Max detections", 10, 300, 100, 10)

    st.markdown("---")

    st.markdown(
        f'<div class="section-title">{icon("lightbulb", "1.3rem")} Did you know?</div>',
        unsafe_allow_html=True,
    )

    tips = [
        ("YOLO means “You Only Look Once”",
         "It detects every object in a single pass through the network — that’s why it’s so fast."),
        ("COCO has 80 everyday classes",
         "From person, car and dog to toothbrush and hair drier. It is the most popular detection benchmark."),
        ("YOLO11 is the latest generation",
         "Released by Ultralytics in 2024, it is more accurate and faster than YOLOv8."),
        ("Confidence is not probability",
         "The percentage you see is the model’s internal confidence score, not a true calibrated probability."),
        ("NMS removes duplicate boxes",
         "Non-Maximum Suppression keeps only the best box when several overlap on the same object."),
    ]

    for title, text in tips:
        st.markdown(
            f"""
            <div class="tip-card">
                <strong>{title}</strong><br>
                {text}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        f'<div class="section-title">{icon("library_books", "1.3rem")} COCO class library</div>',
        unsafe_allow_html=True,
    )

    try:
        _model = load_model("yolo11m.pt")
        COCO_CLASSES = list(_model.names.values())
        st.caption(", ".join(COCO_CLASSES))
    except Exception:
        st.caption("Model not loaded yet.")


# =============================================================================
# Load model
# =============================================================================
with st.spinner("Loading model..."):
    try:
        model = load_model(model_choice)
        model_loaded = True
    except Exception as e:
        model_loaded = False
        st.error(f"Couldn't load model: {e}")

if not model_loaded:
    st.stop()


# =============================================================================
# Hero
# =============================================================================
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-icon">{icon("visibility", "2.1rem")}</div>
        <div>
            <div class="hero-title">Nexus Vision</div>
            <div class="hero-sub">AI-powered object detection</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Image input
# =============================================================================
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
            image = Image.open(uploaded_file)
            image = ImageOps.exif_transpose(image)   # Fix rotation
            image = image.convert("RGB")
        except Exception as e:
            st.error(f"Could not read the uploaded file: {e}")
else:
    camera_file = st.camera_input("Take a picture", label_visibility="collapsed")
    if camera_file is not None:
        try:
            image = Image.open(camera_file)
            image = ImageOps.exif_transpose(image)   # Fix rotation
            image = image.convert("RGB")
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
                model, image, conf_threshold, iou_threshold, max_det, device
            )
        except Exception as e:
            st.error(f"Detection failed: {e}")
            st.stop()

    with col_detected:
        st.markdown(
            f'<div class="section-title">{icon("target", "1.2rem")} Detected</div>',
            unsafe_allow_html=True,
        )
        st.image(annotated_rgb, use_container_width=True)
        st.caption(f"⏱ {inference_time:.2f}s · {device.upper()} · {model_choice}")

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
        st.warning("No objects detected. Try lowering the confidence threshold in the sidebar.")
        st.caption(
            "Tip: YOLO11 can struggle with drawings, cartoons, heavy motion blur, and very low-light images."
        )
else:
    st.markdown(
        f"""
        <div class="card" style="text-align:center; padding: 48px 24px;">
            {icon("upload_file", "2.8rem")}
            <div style="margin-top: 14px; color: #94a3b8; font-size: 1.05rem;">
                Upload an image or use your camera to get started.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# Copyright
# =============================================================================
st.markdown(
    '<div class="copyright">© 2026 Justine Umutoni</div>',
    unsafe_allow_html=True,
)