import streamlit as st
import cv2, torch, tempfile, threading, json, time
from ultralytics import YOLO
from utils.preprocess import blur_sensitive_regions
from utils.gps_utils import get_current_gps
from utils.alert import alert_user
from utils.tracker import ObjectTracker
from utils.map_utils import render_map
from streamlit_folium import st_folium

# Load models
blur_model = YOLO("weights/blur_model.pt")
pothole_model = YOLO("weights/pothole_model.pt")
tracker = ObjectTracker()

st.set_page_config(page_title="DriveWachter", layout="wide")
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
    map_placeholder = st.empty()
    hazards = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: Blur sensitive areas
        frame = blur_sensitive_regions(frame, blur_model)

        # Step 2: Detect potholes
        results = pothole_model(frame, verbose=False)
        annotated_frame = results[0].plot()
        detections = results[0].boxes.xyxy.cpu().numpy()

        # Step 3: Track & alert
        ids = tracker.update(detections)
        gps_data = get_current_gps()

        # Step 4: Log hazard
        if len(ids) > 0 and gps_data['latitude']:
            alert_user("⚠️ Pothole ahead!")
            for pothole_id in ids:
                hazards.append({
                    'id': pothole_id,
                    'lat': gps_data['latitude'],
                    'lon': gps_data['longitude']
                })

        # Step 5: Update Streamlit UI
        stframe.image(annotated_frame, channels="BGR", use_column_width=True)

        if gps_data['latitude']:
            fmap = render_map(gps_data['latitude'], gps_data['longitude'], hazards)
            st_folium(fmap, width=700, height=450)

        time.sleep(0.05)

    cap.release()
