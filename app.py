import streamlit as st
from inference_sdk import InferenceHTTPClient
import cv2
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import numpy as np

# ----------------------
# Initialize Roboflow Client
# ----------------------
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="iCUnknIN3Y51KFSzRiSw"
)

st.set_page_config(page_title="Disabled Person Detector", layout="wide")
st.title("♿ Real-time Disabled Person Detector")
st.markdown("Live detection of disabled persons from your webcam.")

# alert sound file
alert_sound = "alert.wav"

# Draw bounding boxes on frame
def draw_boxes(frame, predictions):
    for pred in predictions:
        x, y = pred["x"], pred["y"]
        w, h = pred["width"], pred["height"]
        cls = pred["class"]
        conf = pred["confidence"]

        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"{cls} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    return frame

# Callback function for live video
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    cv2.imwrite("frame.jpg", img)  # save frame temporarily

    # Inference from Roboflow
    result = CLIENT.infer("frame.jpg", model_id="disabled-person-pkgbq/2")

    # Draw bounding boxes
    img = draw_boxes(img, result["predictions"])

    # Set session state for alert
    st.session_state["detected"] = len(result["predictions"]) > 0

    return img

# Run webcam live streamer
webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
)

# Sound / alert if detected
if st.session_state.get("detected", False):
    st.error("🚨 Disabled Person Detected!")
    audio_file = open(alert_sound, "rb")
    st.audio(audio_file, format="audio/wav")
