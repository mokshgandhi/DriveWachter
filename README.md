# DriveWächter

 Prototype developed for Volkswagen i.mobilothon 5.0
 Team: DriveWächter
 Authors: Moksh Gandhi & Trilok Bhavsar


DriveWächter (“Road Guardian” in German) is a real-time road hazard detection system that identifies potholes using deep-learning segmentation, blurs sensitive visual information, and automatically logs hazard locations using geolocation. Built as a functional prototype for Volkswagen i.mobilothon 5.0, it demonstrates how computer vision can enhance road safety and support driver-assistance systems.

This repository contains the full implementation of the DriveWächter prototype.

# Overview
DriveWächter processes a live video stream (webcam or uploaded file), detects potholes using a YOLOv11 segmentation model, overlays smooth polygon masks based on severity, and logs hazard locations onto an interactive map. Sensitive information—such as license plates and faces—is blurred automatically.

The project is optimized for real-time performance through a streamlined Streamlit interface.

# Features
- Real-Time Pothole Segmentation
- Uses a custom YOLOv11 instance segmentation model (https://app.roboflow.com/drivewachter/potholewachter-yjrxy/models)
- Visualizes potholes with smooth, translucent polygon masks
- Displays minimal severity labels with confidence scores
- Severity Classification

The segmentation model uses the following classes:

| **Class ID** | **Type**       | **Color** |
| ------------ | -------------- | --------- |
| 0            | Medium Pothole | Blue      |
| 1            | Risk Pothole   | Red       |
| 2            | Safe Pothole   | Yellow    |


# Blurring Sensitive Information
Powered by a custom-trained model (PrivacyGuard https://app.roboflow.com/drivewachter/privacyguard-zo4qz/models) to blur:
- Vehicle license plates
- Faces
- Any identifiable sensitive region

# GPS-Based Hazard Logging
- Uses IP-based geolocation
- Stores coordinates, timestamps, and severity
- Renders hazards on a real-time map using Folium

# Voice Alerts
- Audio alerts for new hazard detections
- Cooldown mechanism avoids repeated notifications

# Lightweight Interface
- Built with Streamlit for fast deployment and clean UI
- Real-time updates without blocking

# Roboflow Datasets & Models:

This project uses two custom-trained models hosted on Roboflow:

- 1. PotholeWächter – Pothole Segmentation Model
Dataset + Model:
https://app.roboflow.com/drivewachter/potholewachter-yjrxy/models

- 2. PrivacyGuard – Sensitive Information Blur Model
Dataset + Model:
https://app.roboflow.com/drivewachter/privacyguard-zo4qz/models

- Roboflow Workspace (DriveWächter)
All datasets and experiments:
https://app.roboflow.com/drivewachter

These models were trained specifically for the DriveWächter project.

# Project Structure

DriveWachter/
│
├── weights/
│   ├── blur_model.pt
│   └── pothole_model.pt
│
├── utils/
│   ├── preprocess.py
│   ├── gps_utils.py
│   ├── map_utils.py
│   └── alert.py
│
├── logs/
│   └── session_logs.jsonl
│
├── app.py
└── README.md


# Installation
Clone the Repository
git clone https://github.com/<your-username>/DriveWachter
cd DriveWachter

# Install Dependencies
pip install -r requirements.txt

# Run the Application
streamlit run app.py

# How It Works
1. Frame Processing
Live feed → resizing → sensitive-region blur → segmentation model inference.

2. Pothole Segmentation
YOLOv11 segmentation generates masks → converted to smooth polygons → rendered with translucent severity colors.

3. Minimal Labeling
Each pothole shows a compact label (type + confidence) placed at mask centroid.

4. Hazard Logging
Every unique detection generates:
- One voice alert (cooldown-controlled)
- GPS logging
- Hazard pin on real-time map

5. Map Rendering
Folium map shows:
- Current location
- All logged hazards with timestamps

# Logging

Session logs are stored in:
logs/session_logs.jsonl

They include timestamps, detection counts, and internal tracking metadata.

# Requirements

Python 3.9+
Streamlit
OpenCV
NumPy
Ultralytics YOLO
Geocoder
Webcam (optional)

# Limitations

IP-based GPS is approximate
Accuracy varies by model training quality
Prototype-level performance; not a production ADAS system
Works best with clear, well-lit video inputs

# Future Improvements

- Precise GPS via browser geolocation
- Smartphone-based deployment
- Cloud-based map aggregation
- Depth estimation for severity scoring by integrating LiDAR sensor in bottom for validation
- Lane detection and additional road-condition analytics

# Contributors
Team: DriveWächter
Moksh Gandhi
Trilok Bhavsar
-

Prototype developed for Volkswagen i.mobilothon 5.0
