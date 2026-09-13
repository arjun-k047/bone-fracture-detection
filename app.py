import base64
import os
import shutil
import uuid
import zipfile

import cv2
import numpy as np
import yaml
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "best.pt")
model = YOLO(MODEL_PATH)

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


def preprocess(img_bgr):
    # grayscale + CLAHE before feeding the model, same as what was used
    # during training. clip limit / tile size were picked by eye, didn't
    # do a proper sweep on these.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def allowed_image(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"png", "jpg", "jpeg"}


@app.route("/", methods=["GET", "POST"])
def index():
    result_img = None

    if request.method == "POST":
        file = request.files["image"]
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        pre = preprocess(img)
        results = model(pre)
        plotted = results[0].plot()

        _, buf = cv2.imencode(".jpg", plotted)
        result_img = base64.b64encode(buf).decode("ascii")

    return render_template("index.html", result_img=result_img)


@app.route("/batch")
def batch_page():
    return render_template("batch.html")


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    files = request.files.getlist("images")
    if not files or files[0].filename == "":
        return jsonify({"error": "no files uploaded"}), 400

    results_out = []
    for file in files:
        if not allowed_image(file.filename):
            continue

        filename = secure_filename(file.filename)
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        pre = preprocess(img)

        prediction = model(pre)[0]
        plotted = prediction.plot()

        output_name = f"{uuid.uuid4()}_{filename}"
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, output_name), plotted)

        boxes = []
        for box in prediction.boxes:
            boxes.append({
                "class": prediction.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 4),
            })

        results_out.append({
            "filename": filename,
            "output_url": f"/output/{output_name}",
            "detections": boxes,
        })

    return jsonify({"results": results_out})


@app.route("/output/<filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


@app.route("/evaluate")
def evaluate_page():
    return render_template("evaluate.html")


@app.route("/run_evaluation", methods=["POST"])
def run_evaluation():
    if "images_zip" not in request.files or "labels_zip" not in request.files:
        return jsonify({"error": "upload both an images zip and a labels zip"}), 400

    images_zip = request.files["images_zip"]
    labels_zip = request.files["labels_zip"]

    eval_id = str(uuid.uuid4())
    eval_dir = os.path.join(UPLOAD_FOLDER, f"eval_{eval_id}")
    images_dir = os.path.join(eval_dir, "images")
    labels_dir = os.path.join(eval_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    def extract_flat(zip_file, target_dir):
        with zipfile.ZipFile(zip_file) as zf:
            for info in zf.infolist():
                if info.is_dir() or info.filename.startswith("__MACOSX"):
                    continue
                name = os.path.basename(info.filename)
                if not name:
                    continue
                with zf.open(info) as src, open(os.path.join(target_dir, name), "wb") as dst:
                    shutil.copyfileobj(src, dst)

    try:
        extract_flat(images_zip, images_dir)
        extract_flat(labels_zip, labels_dir)
    except zipfile.BadZipFile:
        return jsonify({"error": "one of the uploaded files isn't a valid zip"}), 400

    dataset_yaml = os.path.join(eval_dir, "dataset.yaml")
    with open(dataset_yaml, "w") as f:
        yaml.dump({
            "path": os.path.abspath(eval_dir),
            "train": "images",
            "val": "images",
            "names": model.names,
        }, f)

    val_dir = os.path.abspath(os.path.join(OUTPUT_FOLDER, f"val_{eval_id}"))
    try:
        metrics = model.val(data=dataset_yaml, project=val_dir, name="result")
    except Exception as e:
        return jsonify({"error": f"validation failed: {e}"}), 500

    return jsonify({
        "eval_id": eval_id,
        "metrics": {
            "map50": round(float(metrics.box.map50), 4),
            "map50-95": round(float(metrics.box.map), 4),
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
        },
    })


@app.route("/eval_output/<eval_id>/<filename>")
def serve_eval_output(eval_id, filename):
    return send_from_directory(os.path.join(OUTPUT_FOLDER, f"val_{eval_id}", "result"), filename)


if __name__ == "__main__":
    app.run(debug=True)
