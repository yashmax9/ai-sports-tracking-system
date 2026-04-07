import numpy as np
from sklearn.cluster import KMeans

colors = []
kmeans = None

def extract_color(frame, bbox):
    x, y, w, h = map(int, bbox)
    crop = frame[y:y+h, x:x+w]

    if crop.size == 0:
        return None

    return crop.mean(axis=(0,1))

def collect_color(color):
    if color is not None:
        colors.append(color)

def train_kmeans():
    global kmeans
    if len(colors) > 20:
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(colors)

def predict_team(color):
    if kmeans is None or color is None:
        return -1
    return int(kmeans.predict([color])[0])