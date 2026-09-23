# 🔍 VisionScan — YOLO Object Detection

A real-time object detection web app built with **YOLOv8** (pretrained on **COCO**) and **Streamlit**. Upload or capture an image and the model draws bounding boxes around detected objects.

## Features

- **80 COCO classes** detected out of the box — no training needed
- **Upload or capture**: choose from file upload or live camera
- **Adjustable confidence & IoU thresholds**: sliders to fine-tune detection sensitivity
- **Confidence explanation**: sidebar note clarifies what the scores mean
- **Detection summary**: counts and average confidence per class
- **Downloadable results**: save the annotated image as a PNG
- **View all COCO classes**: expandable list in the sidebar
- **Deployable to Streamlit Cloud**

## COCO Dataset

**COCO** (Common Objects in Context) is a standard benchmark dataset with **80 object classes**, including people, vehicles, animals, furniture, and everyday household items.

## Tech Stack

- **YOLOv8** (Ultralytics) — object detection
- **Streamlit** — web app framework
- **OpenCV / Pillow / NumPy** — image processing

## Project Structure

```
vision-scan-app/
├── app.py
├── requirements.txt
└── README.md
```

(`yolov8n.pt` downloads automatically the first time you run the app, or you can fetch it ahead of time — see below.)

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/vision-scan-app.git
cd vision-scan-app

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Pre-download the YOLOv8 model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# 5. Launch the app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"**, select your repo, and set the main file to `app.py`.
4. Click **Deploy**.

Your app will be live at `https://your-app-name.streamlit.app` within 1–2 minutes. Ultralytics will download `yolov8n.pt` automatically on first run, so you don't need to commit the weights file to the repo — just make sure the deployment has internet access on first boot.

## About Confidence Scores

The percentage shown next to each detection is a **confidence score**, not a calibrated probability. It reflects the model's certainty about a detection (objectness × class probability). A score of 0.94 means the model is much more confident than one scored 0.70 — not that it is literally 94% likely to be correct.

## Ideas for Extending This Project

- Swap `yolov8n.pt` for a larger variant (`yolov8s.pt`, `yolov8m.pt`) for higher accuracy at the cost of speed
- Fine-tune on a custom dataset for a domain-specific use case (e.g. medical imaging, agriculture)
- Add video/webcam streaming support instead of single-frame capture
- Log detections to a database for analytics over time

## License

Apache 2.0 — free to use for learning and portfolio projects.

## Credits

- Model: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Dataset: [COCO](https://cocodataset.org) (80 classes)
- Framework: [Streamlit](https://streamlit.io)
