# app.py
import streamlit as st
import cv2
import time
import tempfile
import json
import threading
import os
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import folium
from streamlit_folium import st_folium
import geocoder
import pyttsx3

# --------------------------
# Configuration
# --------------------------
WEIGHTS_DIR = Path("weights")
BLUR_MODEL_PATH = WEIGHTS_DIR / "blur_model.pt"
POTHOLE_MODEL_PATH = WEIGHTS_DIR / "pothole_model.pt"
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "session_logs.jsonl"

# TTS engine (non-blocking wrapper)
_tts_engine = None
def _init_tts():
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = pyttsx3.init()
        # reduce volume/speed if needed
        _tts_engine.setProperty("rate", 150)
        _tts_engine.setProperty("volume", 0.9)
    return _tts_engine

def speak_nonblocking(text):
    """Play TTS in a separate thread so UI isn't blocked."""
    def _s():
        e = _init_tts()
        e.say(text)
        e.runAndWait()
    t = threading.Thread(target=_s, daemon=True)
    t.start()

# --------------------------
# Lightweight SORT-like tracker
# --------------------------
class SimpleTracker:
    def __init__(self, max_lost=10, distance_threshold=60):
        self.next_id = 1
        self.max_lost = max_lost
        self.distance_threshold = distance_threshold
        self.objects = {}   # id -> (cx, cy)
        self.lost = {}      # id -> lost_count

    def update(self, detections):
        """
        detections: list of [x1,y1,x2,y2,score,cls]
        returns dict: track_id -> (cx,cy,bbox)
        """
        # compute centers
        det_centers = []
        for det in detections:
            x1,y1,x2,y2,score,cls = det
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            det_centers.append((cx, cy, det))

        assigned = {}
        used_det_idx = set()
        # try to assign existing objects to detections
        for oid, (ox, oy) in list(self.objects.items()):
            best_i = None
            best_d = None
            for i, (cx, cy, det) in enumerate(det_centers):
                if i in used_det_idx:
                    continue
                d = np.hypot(ox - cx, oy - cy)
                if best_d is None or d < best_d:
                    best_d, best_i = d, i
            if best_i is not None and best_d <= self.distance_threshold:
                # assign
                cx, cy, det = det_centers[best_i]
                assigned[oid] = (cx, cy, det)
                used_det_idx.add(best_i)
                self.lost[oid] = 0
            else:
                # not matched this frame
                self.lost[oid] = self.lost.get(oid, 0) + 1

        # remove lost objects
        for oid in list(self.objects.keys()):
            if self.lost.get(oid, 0) > self.max_lost:
                self.objects.pop(oid, None)
                self.lost.pop(oid, None)

        # create new tracks for unassigned detections
        for i, (cx, cy, det) in enumerate(det_centers):
            if i in used_det_idx:
                continue
            new_id = self.next_id
            self.next_id += 1
            assigned[new_id] = (cx, cy, det)
            self.lost[new_id] = 0

        # update stored positions
        new_objects = {}
        for tid, (cx, cy, det) in assigned.items():
            new_objects[tid] = (cx, cy)
        self.objects = new_objects

        # return mapping tid -> (cx,cy,bbox)
        out = {}
        for tid, (cx, cy, det) in assigned.items():
            x1,y1,x2,y2,score,cls = det
            out[tid] = (cx, cy, (int(x1),int(y1),int(x2),int(y2)), float(score), int(cls))
        return out

# --------------------------
# Utilities: GPS, map, logging
# --------------------------
def get_current_gps():
    """Try to get approximate location via IP. Returns (lat, lon) or (None, None)."""
    try:
        g = geocoder.ip('me')
        if g and g.latlng:
            return float(g.latlng[0]), float(g.latlng[1])
    except Exception:
        pass
    return None, None

def render_folium_map(lat, lon, hazards):
    # center map around current location (fallback coordinates if missing)
    center = [lat or 23.0225, lon or 72.5714]
    fmap = folium.Map(location=center, zoom_start=16)
    # current position marker
    folium.Marker(center, tooltip="Current Position", icon=folium.Icon(color='blue', icon='car', prefix='fa')).add_to(fmap)
    # hazard markers
    for h in hazards:
        folium.Marker([h['lat'], h['lon']],
                      popup=f"Pothole ID {h['id']}\n{h.get('label','')}",
                      icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa')).add_to(fmap)
    return fmap

def append_log(entry):
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        # don't crash UI for logging problems
        st.error(f"Failed to write log: {e}")

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="DriveWachter", layout="wide")
st.title("DriveWachter")

# sidebar controls
st.sidebar.header("Settings")
use_webcam = st.sidebar.checkbox("Use webcam (local)", value=True)
upload_file = st.sidebar.file_uploader("Or upload a video (mp4/avi)", type=["mp4","avi","mov"])
display_fps = st.sidebar.checkbox("Show FPS", value=True)
conf_threshold = st.sidebar.slider("Detection confidence threshold", min_value=0.1, max_value=0.9, value=0.45, step=0.05)

# model load (lazy)
if "models_loaded" not in st.session_state:
    st.session_state["models_loaded"] = False
    st.session_state["hazards"] = []  # persistent hazards list
    st.session_state["running"] = False
    st.session_state["last_alerted_ids"] = set()

if not st.session_state["models_loaded"]:
    with st.spinner("Loading models (this may take a few seconds)..."):
        if not BLUR_MODEL_PATH.exists():
            st.error(f"Blur model not found at {BLUR_MODEL_PATH}. Put blur .pt into weights/")
        if not POTHOLE_MODEL_PATH.exists():
            st.error(f"Pothole model not found at {POTHOLE_MODEL_PATH}. Put pothole .pt into weights/")
        try:
            model_blur = YOLO(str(BLUR_MODEL_PATH))
            model_pothole = YOLO(str(POTHOLE_MODEL_PATH))
            st.session_state["model_blur"] = model_blur
            st.session_state["model_pothole"] = model_pothole
            st.session_state["models_loaded"] = True
            st.success("Models loaded ✅")
        except Exception as e:
            st.error(f"Failed loading models: {e}")
            st.stop()

# control buttons
col1, col2, col3 = st.columns([1,1,1])
with col1:
    if st.button("Start"):
        st.session_state["running"] = True
with col2:
    if st.button("Stop"):
        st.session_state["running"] = False
with col3:
    if st.button("Clear hazards"):
        st.session_state["hazards"] = []
        st.session_state["last_alerted_ids"] = set()

# placeholders for UI
video_col, map_col = st.columns([2,1])
video_slot = video_col.empty()
map_slot = map_col.empty()
info_slot = st.sidebar.empty()

# prepare capture source
cap = None
if use_webcam:
    cap_source = 0
elif upload_file is not None:
    # save to temp file
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tf.write(upload_file.read())
    tf.flush()
    cap_source = tf.name
else:
    cap_source = None

# tracker
if "tracker" not in st.session_state:
    st.session_state["tracker"] = SimpleTracker()
    st.session_state["start_time"] = None
    st.session_state["frame_count"] = 0

# main loop
if st.session_state["running"]:
    # open video capture
    if cap_source is None:
        st.warning("No input source selected. Please enable webcam or upload a video.")
        st.session_state["running"] = False
    else:
        cap = cv2.VideoCapture(cap_source)
        # attempt to set FPS on webcam
        try:
            cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass

    if cap is not None and cap.isOpened():
        model_blur = st.session_state["model_blur"]
        model_pothole = st.session_state["model_pothole"]
        tracker = st.session_state["tracker"]

        # runtime state
        start_time = time.time()
        st.session_state["start_time"] = start_time
        st.session_state["frame_count"] = 0

        # loop until stopped or video ends
        while st.session_state["running"]:
            ret, frame = cap.read()
            if not ret:
                # video ended or webcam not accessible
                st.warning("Frame read failed; stopping.")
                st.session_state["running"] = False
                break

            st.session_state["frame_count"] += 1
            frame_idx = st.session_state["frame_count"]

            # ---- 1) Blur sensitive regions (use smaller imgsz if needed) ----
            try:
                r_blur = model_blur(frame, verbose=False)
                # r_blur[0].boxes.xyxy items may be Tensors; iterate safely
                for b in r_blur[0].boxes.xyxy:
                    coords = [int(x) for x in b.tolist()]
                    x1,y1,x2,y2 = coords
                    # safety bounds
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51,51), 0)
            except Exception as e:
                # continue even if blur fails
                st.sidebar.error(f"Blur model error: {e}")

            # ---- 2) Pothole detection ----
            detections = []  # list of [x1,y1,x2,y2,score,cls]
            try:
                r_p = model_pothole(frame, verbose=False)
                boxes_data = r_p[0].boxes.data.cpu().numpy() if hasattr(r_p[0].boxes, "data") else []
                # boxes_data rows are [x1,y1,x2,y2,score,cls]
                for row in boxes_data:
                    x1,y1,x2,y2,score,cls = row
                    if float(score) >= conf_threshold:
                        detections.append([float(x1), float(y1), float(x2), float(y2), float(score), int(cls)])
                        label = r_p[0].names[int(cls)] if hasattr(r_p[0], "names") else str(int(cls))
                        # draw bbox + label
                        cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 2)
                        cv2.putText(frame, f"{label} {score:.2f}", (int(x1), int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            except Exception as e:
                st.sidebar.error(f"Pothole model error: {e}")

            # ---- 3) Tracking ----
            tracks = tracker.update(detections)  # tid -> (cx,cy,bbox,score,cls)
            # draw track IDs
            for tid, (cx, cy, bbox, score, cls) in tracks.items():
                x1,y1,x2,y2 = bbox
                cv2.putText(frame, f"ID:{tid}", (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            # ---- 4) Alerts & hazard persistence ----
            # Fire audio alert when a new track id appears (avoid spamming)
            new_alert_ids = set(tracks.keys()) - st.session_state.get("last_alerted_ids", set())
            for new_id in new_alert_ids:
                # get gps and log hazard if available
                lat, lon = get_current_gps()
                if lat is not None and lon is not None:
                    st.session_state["hazards"].append({"id": int(new_id), "lat": float(lat), "lon": float(lon), "time": time.time()})
                # speak
                speak_nonblocking("Pothole ahead")
            # update last alerted set (keep it small)
            st.session_state["last_alerted_ids"] = set(tracks.keys())

            # ---- 5) Display and map UI ----
            fps = None
            if display_fps:
                elapsed = time.time() - start_time
                fps = st.session_state["frame_count"] / max(1e-6, elapsed)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # show frame in Streamlit
            video_slot.image(frame, channels="BGR", use_container_width=True)

            # show map in map_slot (update every few frames to save CPU)
            if frame_idx % 5 == 0:
                lat, lon = get_current_gps()
                hazards = st.session_state["hazards"]
                fmap = render_folium_map(lat, lon, hazards)
                map_slot.write(st_folium(fmap, width=400, height=400))

            # ---- 6) Logging per frame (non-blocking) ----
            log_entry = {
                "timestamp": time.time(),
                "frame": int(frame_idx),
                "detections_count": len(detections),
                "tracks": [{ "id": int(tid), "bbox": list(tracks[tid][2]), "score": float(tracks[tid][3]), "class": int(tracks[tid][4]) } for tid in tracks],
                "gps": {"lat": lat, "lon": lon}
            }
            # append log asynchronously so I/O doesn't block critical loop
            threading.Thread(target=append_log, args=(log_entry,), daemon=True).start()

            # small sleep to yield control to Streamlit UI
            time.sleep(0.01)

        # release capture when stopped
        cap.release()
        st.session_state["running"] = False
        st.success("Stopped")
    else:
        st.error("Unable to open video source.")
        st.session_state["running"] = False
else:
    # not running -> still display static map & controls
    lat, lon = get_current_gps()
    fmap = render_folium_map(lat, lon, st.session_state.get("hazards", []))
    map_slot.write(st_folium(fmap, width=400, height=400))
    video_slot.info("Press Start to begin processing (webcam or upload).")

# sidebar info
info_slot.write({
    "models_loaded": st.session_state.get("models_loaded", False),
    "running": st.session_state.get("running", False),
    "frames_processed": st.session_state.get("frame_count", 0)
})
