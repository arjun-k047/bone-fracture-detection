"""Minimal local wrapper: upload an X-ray, run CLAHE+grayscale preprocessing, detect fractures with YOLOv11."""
import io
import os

import cv2
import numpy as np
from flask import Flask, request, render_template_string
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "best.pt")
model = YOLO(MODEL_PATH)

app = Flask(__name__)

PAGE = """
<!doctype html>
<title>Fracture Detection</title>
<h1>Bone Fracture Detection</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=image accept="image/*" required>
  <input type=submit value=Detect>
</form>
{% if result_img %}
  <h2>Result</h2>
  <img src="data:image/jpeg;base64,{{ result_img }}">
{% endif %}
"""


def preprocess(img_bgr):
    """CLAHE + grayscale, standard OpenCV recipe.

    ponytail: exact CLAHE params/pipeline used in training are lost (no
    preprocessing script survived) — clip_limit=2.0, tile=8x8 is a
    reasonable default, retune against a sample training image if one
    turns up.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


@app.route("/", methods=["GET", "POST"])
def index():
    result_img = None
    if request.method == "POST":
        file = request.files["image"]
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        pre = preprocess(img)
        results = model(pre)
        plotted = results[0].plot()  # draws boxes + confidence scores
        ok, buf = cv2.imencode(".jpg", plotted)
        import base64
        result_img = base64.b64encode(buf).decode("ascii")
    return render_template_string(PAGE, result_img=result_img)


if __name__ == "__main__":
    app.run(debug=True)
