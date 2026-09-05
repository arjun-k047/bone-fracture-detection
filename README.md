# Bone Fracture Detection

YOLOv11 (Ultralytics) model fine-tuned on the GRAZPEDWRI-DX pediatric wrist
trauma X-ray dataset, wrapped in a minimal local Flask UI.

## Setup

```
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000, upload an X-ray image, get back the image with
predicted fracture boxes and confidence scores.

## Model

`model/best.pt` — see `REPORT.md` for which training run this came from and why.

## Preprocessing caveat

`app.py` applies grayscale + CLAHE (`clipLimit=2.0`, `tileGridSize=(8,8)`)
before inference, matching what training reportedly used. The exact original
parameters were not recovered (see `REPORT.md`) — treat this as a reasonable
default, not a confirmed match, and retune if detections look off.
