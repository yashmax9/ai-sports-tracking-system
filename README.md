# 🚀 AI Sports Tracking & Analytics System

A **GPU-accelerated real-time sports analytics system** built using computer vision and deep learning.
This project detects, tracks, and analyses players in sports videos, providing **movement insights, team classification, and performance analytics**.

---

## 🎯 Overview

This system combines **YOLOv8 object detection** and **DeepSORT tracking** to create a complete pipeline for analysing player behaviour in real time.

It goes beyond simple detection by adding:

* 📊 Movement analytics
* 🧠 Team classification
* 🔁 Re-identification
* 🔥 Heatmap visualisation

---

## ✨ Features

* ⚡ **Real-time Detection (YOLOv8 + GPU)**
* 🎯 **Multi-object Tracking (DeepSORT)**
* 🧠 **Re-identification System (Histogram-based)**
* 👕 **Team Classification (KMeans clustering on jersey colors)**
* 📏 **Distance Tracking per Player**
* 🏃 **Speed Estimation**
* 📍 **Trajectory Tracking**
* 🔥 **Heatmap Generation**
* ⚠️ **Occlusion Detection**
* 🎥 **Supports Both Video & Live Webcam**

---

## 🧠 System Pipeline

```
Video Input
   ↓
YOLOv8 Detection (GPU)
   ↓
DeepSORT Tracking
   ↓
Re-ID Memory System
   ↓
Team Classification (KMeans)
   ↓
Movement Analysis (Distance + Speed)
   ↓
Heatmap + Occlusion Detection
   ↓
Final Annotated Output
```

---

## 🛠️ Tech Stack

* **Python**
* **OpenCV**
* **PyTorch (GPU)**
* **Ultralytics YOLOv8**
* **DeepSORT**
* **Scikit-learn (KMeans)**

---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/ai-sports-tracking-system.git
cd ai-sports-tracking-system

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

---

## ▶️ Usage

### 🔹 Video Mode

Place your video inside:

```
input/video.mp4
```

Run:

```bash
python main.py
```

---

### 🔹 Real-Time Mode (Webcam)

In `main.py`, change:

```python
MODE = "realtime"
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

## 📊 Output

* Real-time annotated video
* Player IDs, distance, and speed
* Team classification (color-based)
* Heatmap overlay
* Occlusion alerts

---

## ⚡ Performance

* GPU acceleration enabled (CUDA)
* Optimised for real-time processing
* Adjustable frame skipping for performance tuning

---

## ⚠️ Limitations

* Team classification depends on clear jersey color differences
* Re-ID is lightweight (histogram-based, not deep learning)
* Performance varies based on GPU capability

---

## 🔥 Future Improvements

* 🎯 Ball tracking
* 🧠 Deep learning-based team classification
* 📊 Web dashboard (Streamlit)
* 📈 Advanced player analytics (zones, passes, events)

---

## 👨‍💻 Author

**Yash Sharma**

---

## ⭐ Final Note

This project demonstrates a **complete real-time AI pipeline**, combining detection, tracking, and analytics into a modular and scalable system.

---
