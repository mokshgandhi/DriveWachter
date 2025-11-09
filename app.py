import cv2
import time
import tempfile
import threading
import json
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import streamlit as st
import streamlit.components.v1 as components

from utils.preprocess import blur_sensitive_regions
from utils.map_utils import render_map
from utils.gps_utils import get_current_gps
from utils.alert import alert_user

WEIGHTS_DIR = Path("weights")
BLUR_MODEL_PATH = WEIGHTS_DIR / "blur_model.pt"
POTHOLE_MODEL_PATH = WEIGHTS_DIR / "pothole_model.pt"

LOGS_DIR = Path("logs"); LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "session_logs.jsonl"

def speak_nonblocking(text):
    threading.Thread(target=lambda: alert_user(text), daemon=True).start()

def safe_get_gps():
    g = get_current_gps()
    lat = g.get("latitude") if isinstance(g, dict) else None
    lon = g.get("longitude") if isinstance(g, dict) else None
    try:
        if lat is None or lon is None:
            return None, None
        return float(lat), float(lon)
    except (TypeError, ValueError):
        return None, None

def append_log(entry):
    def _w(e):
        try:
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps(e) + "\n")
        except Exception:
            pass
    threading.Thread(target=_w, args=(entry,), daemon=True).start()

# Distance-based tracker
class SimpleTracker:
    def __init__(self, max_lost=12, dist_thresh=60):
        self.next_id = 1
        self.objects = {}   # id -> (cx,cy)
        self.lost = {}      # id -> lost_count
        self.max_lost = max_lost
        self.dist_thresh = dist_thresh

    def update(self, detections):
        det_centers = []
        for det in detections:
            x1,y1,x2,y2,score,cls = det
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            det_centers.append((cx, cy, det))

        assigned, used = {}, set()

        for oid, (ox, oy) in list(self.objects.items()):
            best_i = None; best_d = None
            for i, (cx, cy, det) in enumerate(det_centers):
                if i in used: continue
                d = np.hypot(ox - cx, oy - cy)
                if best_d is None or d < best_d:
                    best_d, best_i = d, i
            if best_i is not None and best_d <= self.dist_thresh:
                cx, cy, det = det_centers[best_i]
                assigned[oid] = (cx, cy, det)
                used.add(best_i)
                self.lost[oid] = 0
            else:
                self.lost[oid] = self.lost.get(oid, 0) + 1

        for oid in list(self.objects.keys()):
            if self.lost.get(oid, 0) > self.max_lost:
                self.objects.pop(oid, None)
                self.lost.pop(oid, None)

        for i, (cx, cy, det) in enumerate(det_centers):
            if i in used: continue
            nid = self.next_id; self.next_id += 1
            assigned[nid] = (cx, cy, det)
            self.lost[nid] = 0

        self.objects = {tid:(assigned[tid][0], assigned[tid][1]) for tid in assigned}

        out = {}
        for tid, (cx, cy, det) in assigned.items():
            x1,y1,x2,y2,score,cls = det
            out[tid] = (cx, cy, (int(x1),int(y1),int(x2),int(y2)), float(score), int(cls))
        return out

# Streamlit UI Setup
st.set_page_config(page_title="DriveWachter", layout="wide")
st.title("DriveWächter")
st.markdown(
    """
    <div style='font-size:18px; color: #555; margin-top:-10px;'>
        An AI-powered road hazard detection system that detects potholes, blurs sensitive informations including license plates and faces, 
        tracks hazards in real-time, and logs GPS-based locations during your drive.
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div style='font-size:14px; color:#777; margin-top:4px;'>
        <i>This project is a prototype demonstration created for 
        <b>Volkswagen i.mobilothon 5.0</b> by 
        <b>Moksh Gandhi</b> and <b>Trilok Bhavsar</b>.</i>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("Settings")
use_webcam = st.sidebar.checkbox("Use webcam (local)", value=True)
uploaded_file = st.sidebar.file_uploader("Or upload a video (mp4/avi/mov)", type=["mp4","avi","mov"])
show_fps = st.sidebar.checkbox("Show FPS", value=True)
conf_threshold = st.sidebar.slider("Detection confidence threshold", 0.1, 0.9, 0.45, 0.05)

video_col, map_col = st.columns([2, 1])
video_slot = video_col.empty()
map_slot = map_col.empty()

# Session init
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.models_loaded = False
    st.session_state.running = False
    st.session_state.hazards = []
    st.session_state.last_alerted_ids = set()  # kept but unused
    st.session_state.tracker = SimpleTracker()
    st.session_state.frame_count = 0
    st.session_state.cap = None
    st.session_state.temp_path = None
    st.session_state.map_update_interval_frames = 5
    st.session_state.last_alert_time = 0.0  # cooldown for voice+log

# Load models once
if not st.session_state.models_loaded:
    with st.spinner("Loading YOLOv11 models..."):
        try:
            st.session_state.model_blur = YOLO(str(BLUR_MODEL_PATH))
            st.session_state.model_pothole = YOLO(str(POTHOLE_MODEL_PATH))
            st.session_state.models_loaded = True
        except Exception as e:
            st.error(f"Failed to load models: {e}")
            st.stop()

# Controls
col1, col2, col3 = st.columns([1,1,1])
with col1:  start_btn = st.button("Start")
with col2:  stop_btn  = st.button("Stop")
with col3:  clear_btn = st.button("Clear hazards")

if clear_btn:
    st.session_state.hazards.clear()
    st.session_state.last_alerted_ids.clear()

# Choose source
cap_source = None
if use_webcam:
    cap_source = 0
elif uploaded_file is not None:
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tf.write(uploaded_file.read())
    tf.close()
    st.session_state.temp_path = tf.name
    cap_source = st.session_state.temp_path

# Start / Stop flags
if start_btn and not st.session_state.running:
    if cap_source is None:
        st.warning("Select webcam or upload a video first.")
    else:
        st.session_state.running = True
        st.session_state.frame_count = 0

if stop_btn and st.session_state.running:
    st.session_state.running = False

# Main-thread processing loop
def render_map_safe():
    lat, lon = safe_get_gps()
    if lat is not None and lon is not None:
        try:
            fmap = render_map(lat, lon, st.session_state.hazards)
            components.html(fmap._repr_html_(), height=420)
        except Exception:
            map_slot.info("Map unavailable.")
    else:
        map_slot.info("Map waiting for GPS — enable location/network access.")

def open_capture_if_needed(source):
    if st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(source)
        time.sleep(0.05)
    return st.session_state.cap

def close_capture():
    try:
        if st.session_state.cap is not None:
            st.session_state.cap.release()
    except Exception:
        pass
    st.session_state.cap = None

# If running, process frames in this run
if st.session_state.running:
    cap = open_capture_if_needed(cap_source)
    tracker = st.session_state.tracker
    model_blur = st.session_state.model_blur
    model_pothole = st.session_state.model_pothole

    start_time = time.time()

    # Process until user hits Stop or file ends
    # Keep this loop responsive but UI-safe (no background UI calls)
    while st.session_state.running:
        ret, frame = cap.read()

        if not ret:
            st.session_state.running = False
            break

        st.session_state.frame_count += 1
        idx = st.session_state.frame_count

        # Resize for speed
        frame = cv2.resize(frame, (640, 384))

        # 1) Blur sensitive
        try:
            frame = blur_sensitive_regions(frame, model_blur)
        except Exception:
            pass
        
        # 2) Segmentation-based pothole detection
        detections = []
        try:
            r_p = model_pothole(frame, verbose=False)
            result = r_p[0]

            if hasattr(result, "masks") and result.masks is not None:
                masks = result.masks.data.cpu().numpy()      # [N,H,W]
                boxes = result.boxes.data.cpu().numpy()      # [N,6]
                names = result.names

                COLORS = {
                    0: (255, 0, 0),     # medium → blue
                    1: (0, 0, 255),     # risk   → red
                    2: (0, 255, 255),   # safe   → yellow
                }

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2, score, cls = box
                    score = float(score)
                    cls = int(cls)

                    if score < conf_threshold:
                        continue

                    detections.append([float(x1), float(y1), float(x2), float(y2), score, cls])

                    mask = masks[i].astype(np.uint8)

                    # Extract contours from mask
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    color = COLORS.get(cls, (0, 255, 255))  # default to yellow if unknown

                    # Translucent fill via overlay
                    overlay = frame.copy()
                    for cnt in contours:
                        if cv2.contourArea(cnt) < 60:
                            continue
                        epsilon = 0.005 * cv2.arcLength(cnt, True)
                        smooth_cnt = cv2.approxPolyDP(cnt, epsilon, True)
                        cv2.fillPoly(overlay, [smooth_cnt], color)
                    frame = cv2.addWeighted(overlay, 0.33, frame, 0.67, 0)

                    for cnt in contours:
                        if cv2.contourArea(cnt) < 60:
                            continue
                        epsilon = 0.005 * cv2.arcLength(cnt, True)
                        smooth_cnt = cv2.approxPolyDP(cnt, epsilon, True)
                        cv2.polylines(frame, [smooth_cnt], True, color, 1, lineType=cv2.LINE_AA)

                    M = cv2.moments(mask)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        label_name = names[cls] if hasattr(result, 'names') else str(cls)
                        label = f"{label_name} {score*100:.1f}%"
                        cv2.putText(
                            frame, label, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (255,255,255), 1, cv2.LINE_AA
                        )

        except Exception as e:
            print("Segmentation error:", e)

        tracks = tracker.update(detections)

        # 4) Hazard alert with cooldown (no ID logic)
        if len(detections) > 0:
            now = time.time()
            if now - st.session_state.last_alert_time > 2.0:
                #alert only when a risk-pothole exists
                if any(d[5] == 1 for d in detections):   #class 1 = risk
                    speak_nonblocking("Pothole ahead")
                    st.session_state.last_alert_time = now

                # Log hazard once per alert with GPS if available
                lat, lon = safe_get_gps()
                if lat is not None and lon is not None:
                    st.session_state.hazards.append({
                        "lat": float(lat),
                        "lon": float(lon),
                        "time": now
                    })

        # 5) FPS
        if show_fps:
            elapsed = max(1e-6, time.time() - start_time)
            fps = idx / elapsed
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 6) Update video
        video_slot.image(frame, channels="BGR", use_container_width=True)

        # 7) Update map occasionally
        if idx % st.session_state.map_update_interval_frames == 0:
            map_slot.empty()
            render_map_safe()

        # 8) Async log write
        append_log({
            "timestamp": time.time(),
            "frame": int(idx),
            "detections_count": len(detections),
            "tracks": [
                {"id": int(tid), "bbox": list(tracks[tid][2]), "score": float(tracks[tid][3]), "class": int(tracks[tid][4])}
                for tid in tracks
            ],
        })

        time.sleep(0.005)

    close_capture()
    if st.session_state.temp_path and Path(st.session_state.temp_path).exists():
        pass

else:
    map_slot.empty()
    render_map_safe()

st.sidebar.write({
    "models_loaded": st.session_state.get("models_loaded", False),
    "running": st.session_state.get("running", False),
    "frames_processed": st.session_state.get("frame_count", 0),
    "hazards_logged": len(st.session_state.get("hazards", []))
})
