from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.to("cuda")  # GPU

def detect(frame):
    results = model(frame, verbose=False)[0]
    detections = []

    for r in results.boxes.data:
        x1, y1, x2, y2, conf, cls = r.tolist()

        if int(cls) == 0 and conf > 0.5:
            detections.append(([x1, y1, x2-x1, y2-y1], conf, 'person'))

    return detections