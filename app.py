import streamlit as st
from inference_sdk import InferenceHTTPClient
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import cv2
import numpy as np
import tempfile
import threading

# Roboflow Client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="iCUnknIN3Y51KFSzRiSw"
)

st.set_page_config(page_title="Disabled Person Detector", layout="wide")
st.title("♿ Live Disabled Person Detector")
st.markdown("The camera will detect disabled persons in real-time.")

# Sound alert file
alert_sound = "alert.wav"

# Session state to avoid multiple alerts at the same time
if "alert_playing" not in st.session_state:
    st.session_state["alert_playing"] = False

# Draw bounding boxes function
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

# Play alert sound in background
def play_alert():
    if not st.session_state["alert_playing"]:
        st.session_state["alert_playing"] = True
        audio_file = open(alert_sound, "rb")
        st.audio(audio_file, format="audio/wav")
        st.session_state["alert_playing"] = False

# Video frame callback
def callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    # Save temporarily
    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(temp_file.name, img)

    # Roboflow inference
    result = CLIENT.infer(temp_file.name, model_id="disabled-person-pkgbq/2")

    # Draw boxes
    img = draw_boxes(img, result["predictions"])

    # Play alert if detected
    if len(result["predictions"]) > 0:
        threading.Thread(target=play_alert).start()

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Run live webcam
webrtc_streamer(
    key="disabled_person_detector",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=callback,
    media_stream_constraints={"video": True, "audio": False},
)
