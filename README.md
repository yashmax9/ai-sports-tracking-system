# 🚀 Real-Time Sports Analytics AI System

A **GPU-accelerated real-time sports analytics system** built using computer vision and deep learning.

This project detects, tracks, and analyses players in sports videos, providing **movement insights, team classification, and performance analytics**.

---

## 🎯 Key Features

* ⚡ YOLOv8 **real-time detection (GPU accelerated)**
* 🎯 DeepSORT **multi-object tracking**
* 🧠 Re-identification system (appearance-based)
* 👕 Team classification (jersey color clustering)
* 📏 Distance tracking per player
* 🏃 Speed estimation
* 📍 Trajectory tracking
* 🔥 Heatmap generation
* ⚠️ Occlusion detection
* 🎥 Supports both **video & live webcam**

---

## 🧠 System Pipeline

```
Video Input
   ↓
YOLOv8 Detection (GPU)
   ↓
DeepSORT Tracking
   ↓
Re-ID System
   ↓
Team Classification
   ↓
Movement Analytics (Distance + Speed)
   ↓
Heatmap + Occlusion Detection
   ↓
Final Output
```

## 🛠️ Tech Stack

* Python
* OpenCV
* PyTorch (CUDA)
* YOLOv8 (Ultralytics)
* DeepSORT
* Scikit-learn

---

## ▶️ How to Run

```bash
python main.py
```

---

## ⚙️ Modes

Change in `main.py`:

```
MODE = "video"      # or "realtime"
```

---

## 📁 Project Structure

```
sports-tracking-ai/
│── main.py
│── detector.py
│── tracker.py
│── movement.py
│── reid.py
│── team_classifier.py
│── utils.py
│── input/
│── output/
```

---

## ⚡ Performance

* GPU acceleration enabled
* Real-time processing supported
* Optimised pipeline design

---

## 🎯 Real-World Applications

* Sports analytics & coaching insights
* Player performance tracking
* Match strategy analysis

---

## ⚠️ Limitations

* Team classification depends on jersey color contrast
* Re-ID is lightweight (not deep learning based)

---

## 🔥 Future Improvements

* Ball tracking
* Deep learning-based team detection
* Web dashboard (Streamlit)
* Advanced analytics

---

## 👨‍💻 Author

**Yash Sharma**
