import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
from ultralytics import YOLO
import numpy as np

st.sidebar.title("📄 About This App")
st.sidebar.markdown("""
An interactive object detection app using YOLOv8.

🎯 Click "**Start Detection**" to begin real-time object detection.  
Click "**Stop Detection**" to stop it.

Powered by **LUMI MATE AI**.
""")

st.title("🎯 Button-Controlled Object Detection")
st.markdown("Click '**Start Detection**' to activate real-time object detection using YOLO. Then click '**Stop Detection**' to stop.")

# Load YOLO model once
model = YOLO("yolov8s.pt")

def process_frame(frame):
    img = frame.to_ndarray(format="bgr24")

    results = model(img)[0]
    annotated_img = results.plot()

    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results = model(img)[0]
    annotated_img = results.plot()
    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# Streamlit-webrtc component for webcam streaming and processing
webrtc_ctx = webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    # Only show start/stop buttons automatically
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

# Footer
st.markdown("""
<hr style="margin-top: 50px; margin-bottom: 10px;">
<div style="text-align: center; font-size: 14px; color: grey;">
    Developed by <strong>LUMI MATE AI</strong>
</div>
""", unsafe_allow_html=True)
