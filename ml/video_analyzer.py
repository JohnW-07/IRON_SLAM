import cv2
import json
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # lightweight model

def analyze_video(path):
    cap = cv2.VideoCapture(path)

    total_frames = 0
    object_counts = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        if total_frames % 5 != 0:
            continue  # speed up for hackathon

        results = model(frame)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                object_counts[label] = object_counts.get(label, 0) + 1

    cap.release()

    return json.dumps({
        "total_frames": total_frames,
        "detected_objects": object_counts
    })