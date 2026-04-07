import cv2
import math

positions = {}
distance = {}
trajectories = {}

def update_position(track_id, center):
    trajectories.setdefault(track_id, []).append(center)

    if track_id in positions:
        prev = positions[track_id]
        dist = math.sqrt((center[0]-prev[0])**2 + (center[1]-prev[1])**2)
        distance[track_id] = distance.get(track_id, 0) + dist

    positions[track_id] = center

def get_distance(track_id):
    return int(distance.get(track_id, 0))

def draw_trajectory(frame, track_id):
    if track_id in trajectories:
        pts = trajectories[track_id]
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], (255,0,0), 1)

print("MOVEMENT FILE LOADED")