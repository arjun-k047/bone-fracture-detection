"""
Training script for the fracture detector. Run on Colab with a T4 GPU,
dataset mounted from Drive. Kept here mostly for reference since the
notebook itself is gone.

Expects a YOLO-format dataset (images/ + labels/ + data.yaml) - I used
the GRAZPEDWRI-DX wrist X-ray set exported through Roboflow.
"""
from ultralytics import YOLO

DATA_YAML = "dataset/data.yaml"  # path to your exported dataset

model = YOLO("yolo11m.pt")

model.train(
    data=DATA_YAML,
    epochs=60,
    imgsz=640,
    batch=8,
    optimizer="AdamW",
    lr0=0.001,
    seed=0,
    project="fracture_project",
    name="yolo11m_fracture_final",
)
