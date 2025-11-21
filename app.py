import streamlit as st
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
import time
import threading

st.set_page_config(page_title="Disabled Person Detector", layout="wide")
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        max-width: 1100px;
    }
    .stButton > button { width: 100%; }
</style>
""", unsafe_allow_html=True)
st.markdown("### ♿ Disabled Person Detector", unsafe_allow_html=True)

# Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="iCUnknIN3Y51KFSzRiSw"
)

# Alert sound
ALERT_SOUND = "alert.wav"

# Session state for start/stop camera
if 'run_camera' not in st.session_state:
    st.session_state['run_camera'] = False
if 'alert_playing' not in st.session_state:
    st.session_state['alert_playing'] = False

# Layout columns
controls_col, video_col, prediction_col = st.columns([1,3,2])

with controls_col:
    st.markdown("##### Controls")
    if st.button('Start Camera', key='start'):
        st.session_state['run_camera'] = True
    if st.button('Stop Camera', key='stop'):
        st.session_state['run_camera'] = False

with video_col:
    st.markdown("##### Camera Feed")
    FRAME_WINDOW = st.empty()
    status_text = st.empty()

with prediction_col:
    st.markdown("##### Prediction")
    prediction_text = st.empty()

FEED_WIDTH = 500  # px, adjust as needed

# Draw bounding boxes
def draw_boxes(frame, predictions):
    for pred in predictions:
        x, y = pred["x"], pred["y"]
        w, h = pred["width"], pred["height"]
        cls = pred["class"]
        conf = pred["confidence"]
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{cls} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

# Play alert sound
def play_alert():
    if not st.session_state['alert_playing']:
        st.session_state['alert_playing'] = True
        audio_file = open(ALERT_SOUND, "rb")
        st.audio(audio_file, format="audio/wav")
        st.session_state['alert_playing'] = False

# Camera loop
if st.session_state['run_camera']:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    status_text.info('Camera started. Showing live predictions.')

    while st.session_state['run_camera'] and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            status_text.error('Failed to read from camera.')
            break

        # Save temp frame for inference
        temp_file = "frame.jpg"
        cv2.imwrite(temp_file, frame)

        # Roboflow inference
        result = CLIENT.infer(temp_file, model_id="disabled-person-pkgbq/2")

        # Draw boxes
        frame = draw_boxes(frame, result["predictions"])

        # Prediction text
        if len(result["predictions"]) > 0:
            prediction_text.markdown(f"🚨 Disabled Person Detected!")
            threading.Thread(target=play_alert).start()
        else:
            prediction_text.markdown("No Disabled Person Detected")

        # Show frame
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=FEED_WIDTH)
        time.sleep(0.05)  # ~20 FPS

    cap.release()
    status_text.info('Camera stopped.')
else:
    status_text.warning('Camera is off. Click "Start Camera" to begin.')
    FRAME_WINDOW.empty()
    prediction_text.empty()
