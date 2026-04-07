import cv2
import numpy as np
import time

from detector import detect
from tracker import track
from movement import update_position, get_distance, draw_trajectory
from utils import compute_iou
from reid import get_hist, update_memory
from team_classifier import extract_color, collect_color, train_kmeans, predict_team

# ==========================
# MODE: "video" or "realtime"
# ==========================
MODE = "video"   # 🔥 CHANGE HERE

# ==========================
# VIDEO SOURCE
# ==========================
if MODE == "realtime":
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture("input/video.mp4")

# 🔥 IMPORTANT CHECK
if not cap.isOpened():
    print("❌ ERROR: Cannot open video/webcam")
    exit()
else:
    print("✅ Source opened successfully")

WIDTH, HEIGHT = 640, 360

heatmap = np.zeros((HEIGHT, WIDTH))
frame_count = 0

if MODE == "video":
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output/output.mp4', fourcc, 20.0, (WIDTH, HEIGHT))

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break

    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    frame_count += 1

    detections = detect(frame)
    tracks = track(detections, frame)

    boxes = []

    for t in tracks:
        if not t.is_confirmed():
            continue

        track_id = t.track_id
        l, t_, r, b = map(int, t.to_ltrb())

        w = r - l
        h = b - t_
        center = (l + w//2, t_ + h//2)

        update_position(track_id, center)
        dist = get_distance(track_id)

        # Heatmap
        if 0 <= center[1] < HEIGHT and 0 <= center[0] < WIDTH:
            heatmap[center[1]][center[0]] += 1

        # Re-ID
        hist = get_hist(frame, (l, t_, w, h))
        update_memory(track_id, hist)

        # Team classification
        color_feat = extract_color(frame, (l, t_, w, h))
        collect_color(color_feat)

        if frame_count == 50:
            train_kmeans()

        team = predict_team(color_feat)

        if team == 0:
            box_color = (0,0,255)
        elif team == 1:
            box_color = (255,0,0)
        else:
            box_color = (0,255,0)

        label = f"ID:{track_id} D:{dist}"

        cv2.rectangle(frame, (l, t_), (r, b), box_color, 2)
        cv2.putText(frame, label, (l, t_-10), 0, 0.5, box_color, 2)

        draw_trajectory(frame, track_id)

        boxes.append((l, t_, w, h))

    # Occlusion
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            if compute_iou(boxes[i], boxes[j]) > 0.5:
                cv2.putText(frame, "OCCLUSION", (50,50), 0, 1, (0,0,255), 2)

    # Heatmap overlay
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
    frame = cv2.addWeighted(frame, 0.85, heatmap_color, 0.15, 0)

    # FPS
    curr_time = time.time()
    fps = 1/(curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS:{int(fps)}", (10,20), 0, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"Players:{len(boxes)}", (10,40), 0, 0.6, (255,255,255), 2)

    cv2.imshow("AI Tracking System", frame)

    if MODE == "video":
        out.write(frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()

if MODE == "video":
    out.release()

cv2.destroyAllWindows()