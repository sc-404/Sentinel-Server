from flask import Flask, Response, abort, jsonify
import cv2
import urllib.parse
from ultralytics import YOLO

app = Flask(__name__)

# Testing cam config
RTSP_HOST = "192.168.0.167"
RTSP_PORT = 554
#RTSP_PATH = "/stream1" # tplink
RTSP_PATH = "/Streaming/Channels/101"

RTSP_USERNAME = "admin"
RTSP_PASSWORD = "pass"

model = YOLO("yolov8n.pt")  # nano = lightweight

def build_rtsp_url():
    # NOTE: TP-Link often prefers raw credentials
    user = urllib.parse.quote(RTSP_USERNAME, safe="")
    pwd = urllib.parse.quote(RTSP_PASSWORD, safe="")

    return f"rtsp://{user}:{pwd}@{RTSP_HOST}:{RTSP_PORT}{RTSP_PATH}"

def grab_frame():
    rtsp_url = build_rtsp_url() + "?rtsp_transport=tcp"

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return None

    frame = None
    success = False

    # Warm-up frames
    for _ in range(5):
        success, frame = cap.read()
        if success:
            break

    cap.release()
    return frame if success else None


# -----------------------------
# AI DETECTION
# -----------------------------
def detect_objects(frame):
    results = model(frame, imgsz=640, conf=0.4, verbose=False)
    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            detections.append({
                "label": model.names[cls],
                "confidence": round(float(box.conf[0]), 3),
                "bbox": [
                    int(x) for x in box.xyxy[0].tolist()
                ]  # [x1, y1, x2, y2]
            })

    return detections


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/snapshot")
def snapshot():
    frame = grab_frame()
    if frame is None:
        abort(500, "Failed to read frame")

    ok, png = cv2.imencode(".png", frame)
    if not ok:
        abort(500, "PNG encoding failed")

    return Response(png.tobytes(), mimetype="image/png")


@app.route("/detect")
def detect():
    frame = grab_frame()
    if frame is None:
        abort(500, "Failed to read frame")

    detections = detect_objects(frame)

    return jsonify({
        "detections": detections,
        "count": len(detections)
    })


@app.route("/annotated")
def annotated():
    frame = grab_frame()
    if frame is None:
        abort(500, "Failed to read frame")

    detections = detect_objects(frame)

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        label = f"{d['label']} {d['confidence']:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    ok, png = cv2.imencode(".png", frame)
    if not ok:
        abort(500, "PNG encoding failed")

    return Response(png.tobytes(), mimetype="image/png")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
