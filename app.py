from flask import Flask, render_templates, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

net = cv2.dnn.readNetFromONNX("yolov8n.onnx")

CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant",
    "bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair drier","toothbrush"
]

@app.route("/")
def index():
    return render_templates("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.json["image"]
        img_bytes = base64.b64decode(data.split(",")[1])
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        h, w, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, 1/255.0, (640,640), swapRB=True, crop=False)
        net.setInput(blob)

        preds = net.forward()[0]  # (84, 8400)
        preds = preds.transpose()  # (8400, 84)

        detections = []

        for pred in preds:
            confidence = pred[4]
            if confidence < 0.5:
                continue

            class_scores = pred[5:]
            class_id = np.argmax(class_scores)
            score = class_scores[class_id]

            if score < 0.5:
                continue

            cx, cy, bw, bh = pred[:4]

            x = int((cx - bw/2) * w)
            y = int((cy - bh/2) * h)
            bw = int(bw * w)
            bh = int(bh * h)

            detections.append({
                "label": CLASSES[class_id],
                "x": x,
                "y": y,
                "w": bw,
                "h": bh,
                "desc": f"{CLASSES[class_id]} detected"
            })

        return jsonify({
            "count": len(detections),
            "detections": detections
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
