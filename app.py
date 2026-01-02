from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # fast & lightweight

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files.get("image")
    if not file:
        return jsonify({"count": 0, "description": [], "objects": []})

    # Convert image to OpenCV format
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    # Run detection
    results = model(img, conf=0.4)[0]

    objects = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        objects.append({
            "label": label,
            "box": [x1, y1, x2, y2]
        })

    return jsonify({
        "count": len(objects),
        "description": list(set(obj["label"] for obj in objects)),
        "objects": objects
    })

if __name__ == "__main__":
    app.run(debug=True)
