import cv2

memory = {}

def get_hist(frame, bbox):
    x, y, w, h = map(int, bbox)
    crop = frame[y:y+h, x:x+w]

    if crop.size == 0:
        return None

    hist = cv2.calcHist([crop], [0], None, [256], [0,256])
    return cv2.normalize(hist, hist).flatten()

def update_memory(track_id, hist):
    if hist is not None:
        memory[track_id] = hist

print("REID FILE LOADED")