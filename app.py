import base64
import os

import cv2
import numpy as np
from flask import Flask, render_template, request
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "best.pt")
model = YOLO(MODEL_PATH)

app = Flask(__name__)


def preprocess(img_bgr):
    # grayscale + CLAHE before feeding the model, same as what was used
    # during training. clip limit / tile size were picked by eye, didn't
    # do a proper sweep on these.
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
        plotted = results[0].plot()

        _, buf = cv2.imencode(".jpg", plotted)
        result_img = base64.b64encode(buf).decode("ascii")

    return render_template("index.html", result_img=result_img)


if __name__ == "__main__":
    app.run(debug=True)
