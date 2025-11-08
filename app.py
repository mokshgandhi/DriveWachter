import streamlit as st
import cv2, torch, tempfile, json, threading
from ultralytics import YOLO
from utils.preprocess import blur_sensitive_regions
from utils.gps_utils import get_current_gps
from utils.alert import alert_user
from utils.tracker import ObjectTracker

# Load models
blur_model = YOLO("weights/blur_model.pt")
pothole_model = YOLO("weights/pothole_model.pt")
tracker = ObjectTracker()

st.title("🚘 DriveWachter: Real-time Pothole & Hazard Detection")

source_option = st.radio("Select Input Source", ["Webcam", "Upload Video"])

if source_option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
else:
    video_path = 0

run_btn = st.button("Start Detection")

if run_btn:
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: Blur
        frame = blur_sensitive_regions(frame, blur_model)

        # Step 2: Detect potholes
        results = pothole_model(frame, verbose=False)
        annotated_frame = results[0].plot()

        # Step 3: Track & Alert
        detections = results[0].boxes.xyxy.cpu().numpy()
        ids = tracker.update(detections)
        if len(ids) > 0:
            alert_user()

        # Step 4: GPS Logging
        gps_data = get_current_gps()
        st.json(gps_data)

        stframe.image(annotated_frame, channels="BGR")

    cap.release()

