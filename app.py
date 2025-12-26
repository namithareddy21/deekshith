from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Load ONNX model
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
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.json["image"]
        img_bytes = base64.b64decode(data.split(",")[1])
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        h, w, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1/255, (640,640), swapRB=True)
        net.setInput(blob)
        output = net.forward()[0]

        detections = []

        for det in output:
            conf = det[4]
            if conf > 0.45:
                scores = det[5:]
                class_id = np.argmax(scores)
                if scores[class_id] > 0.45:
                    cx, cy, bw, bh = det[:4]
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
