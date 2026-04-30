# 📄 Agricultural Monitoring System (AI-Powered Smart Farming)

An AI-powered agricultural monitoring platform designed for real-time crop analysis, combining computer vision, machine learning, and field simulation. The system detects plant diseases, identifies weeds, and predicts soil health, while tagging all results with location data.

Built initially as a competition project, this system is being extended toward a scalable startup solution for precision agriculture.

---

## 🚀 Features

### 🌿 Leaf Disease Detection

* Custom-trained YOLO model for detecting infected regions
* Cropped regions classified using a ResNet-34 model trained from scratch
* Dataset:

  * Roboflow dataset
  * Self-collected and labeled images
* Classes:

  * Blast
  * Blight
  * Brown Spot
  * Healthy

**Accuracy: 91%**

This two-stage pipeline improves reliability by separating detection and classification.

---

### 🌱 Weed Detection

* YOLO-based real-time weed detection
* Trained on curated datasets for field conditions
* Outputs bounding boxes and detection logs

---

### 🌍 GPS-Based Field Mapping

* Simulated GPS system for field deployment
* Every detection includes:

  * Latitude
  * Longitude
  * Timestamp

Designed to integrate with rover or drone-based monitoring systems.

---

### 🌾 Soil Health Prediction

* Machine learning model trained on real soil data
* Periodic prediction using live input from CSV
* Runs asynchronously in a background thread

**Accuracy: 97%**

---

### 📹 Live Camera Streaming

* Mobile camera used as input via IP streaming
* Real-time frame processing with OpenCV
* Continuous inference on live feed

---

### 📊 Detection Logging System

* Stores results in CSV format
* Tracks recent detections in memory
* Each record includes:

  * Detection type
  * Prediction
  * GPS coordinates
  * Timestamp

---

### 🖥️ Web Dashboard

* Live video feed interface
* Camera control system
* Real-time detection updates
* Clean and responsive UI

---

## 🧠 System Architecture

```
Camera Input → Flask Server → Frame Processing
                                ↓
              ┌───────────────┬───────────────┐
              │               │               │
        Leaf Detection   Weed Detection   Soil Model
              ↓               ↓               ↓
        ResNet Classifier     │        Real Data Input
              ↓               ↓               ↓
              └──────→ Result Logger ←───────┘
                              ↓
                     Web Dashboard
```

---

## ⚙️ Tech Stack

* Backend: Flask
* Computer Vision: OpenCV
* Deep Learning:

  * YOLO (custom trained)
  * ResNet-34 (custom trained)
* Machine Learning: Scikit-learn
* Dataset Tools: Roboflow
* Frontend: HTML, CSS
* Streaming: MJPEG over HTTP
* Concurrency: Python threading

---

## 📂 Project Structure

```
agri-monitor/
│
├── app.py
├── templates/
│   └── index.html
│
├── leaf_disease.pt
├── weed.pt
├── leaf_resnet.pth
├── health_bundle.pkl
│
├── rand.csv
├── result.csv
│
├── requirements.txt
├── Procfile
├── render.yaml
```

---

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd agri-monitor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
python app.py
```

### 4. Open in browser

```
http://localhost:5000
```

---

## 📱 Camera Setup

1. Install an IP camera app on your phone
2. Connect both devices to the same network
3. Enter the stream URL in the dashboard
4. Start live monitoring

---

## 📈 Future Scope

* Integration with real GPS hardware modules
* Deployment on edge devices such as Jetson Nano
* Automated alert system for farmers
* Cloud-based analytics and storage
* Mobile application for field usage

---

## 🎯 Project Vision

This project aims to evolve into a **precision agriculture platform** that enables:

* Early disease detection
* Efficient weed management
* Data-driven soil monitoring

The long-term goal is to develop a scalable solution that supports farmers with real-time insights and automation.

---

## 📌 Notes

* Models must be placed in the root directory
* Ensure correct CSV input format for soil predictions
* GPU recommended for real-time performance

---

## 📜 License

For academic, research, and prototype development use.

---


