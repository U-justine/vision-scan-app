# 🔍 Nexus Vision — YOLO11 Object Detection

A polished, deep-slate-and-teal object detection web app built with **YOLO11** and **Streamlit** — a glassmorphism hero header, animated stat cards, an educational "Did you know?" sidebar, and a clean detection breakdown with icons throughout.

## Features

- **YOLO11** (Ultralytics' latest generation, faster and more accurate than YOLOv8)
- **Advanced settings panel** — model size, device, confidence, IoU, and max detections, tucked into a collapsible expander so the main flow stays uncluttered
- **Model choice**: nano / small / medium (default) / large, traded off between speed and accuracy
- **Device choice**: CPU, CUDA, or MPS, for whichever hardware you're running on
- **Upload or camera capture**, with automatic downscaling of oversized images
- **"Did you know?" sidebar** — short facts about YOLO, COCO, and confidence scores, plus the full 80-class COCO library
- **Live stats**: object count, unique classes, average confidence, inference time
- **Detection breakdown** with per-class count and min/avg/max confidence
- **Bounding box coordinates** in a collapsible expander for debugging
- **Downloadable annotated image**
- **Defensive error handling** for bad uploads, failed inference, and unknown class IDs

## Tech Stack

- **YOLO11** (Ultralytics) — object detection
- **Streamlit** — web app framework
- **OpenCV / Pillow / NumPy** — image processing

## Project Structure

```
nexus-vision/
├── app.py
├── requirements.txt
└── README.md
```

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/nexus-vision.git
cd nexus-vision

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

Then open `http://localhost:8501`. The default model (`yolo11m.pt`) downloads automatically on first run; switching models in Advanced settings downloads the new weights the first time you select them.

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"**, select your repo, and set the main file to `app.py`.
4. Click **Deploy**.

Ultralytics downloads model weights on first run, so no need to commit `.pt` files — just make sure the deployment has internet access on first boot. Streamlit Cloud's free tier is CPU-only, so leave **Device** set to `cpu`; `yolo11n` or `yolo11s` will feel noticeably snappier there than `yolo11m` or `yolo11l`.

## Tuning for fewer wrong predictions

The confidence and IoU sliders in **Advanced settings** are your main levers:

- **Raise the confidence threshold** (above the 0.40 default) to hide anything the model isn't fairly sure about — the single biggest lever against wrong detections
- **Lower the IoU threshold** to remove more duplicate/overlapping boxes on the same object
- **Use a larger model** (`yolo11m` or `yolo11l`) for better base accuracy, at the cost of speed

No object detector is 100% accurate, and pushing precision up will sometimes cost you a borderline true detection in return. If a specific class keeps misfiring, raising the confidence threshold or switching to a larger model are the most reliable fixes.

## About Confidence Scores

The percentage shown next to each detection is a **confidence score**, not a calibrated probability. It reflects the model's certainty about a detection (objectness × class probability) — a score of 0.94 means the model is much more confident than one scored 0.70, not that it's literally right 94% of the time.

## License

Apache 2.0 — free to use for learning and portfolio projects.

## Credits

- Model: [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- Dataset: [COCO](https://cocodataset.org) (80 classes)
- Framework: [Streamlit](https://streamlit.io)
- Icons: [Material Symbols](https://fonts.google.com/icons)

© 2026 Justine Umutoni