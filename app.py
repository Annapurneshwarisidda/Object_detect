import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
from ultralytics import YOLO
import av
import numpy as np

# ---------------- Sidebar ----------------
st.sidebar.title("📄 About This App")
st.sidebar.markdown("""
An interactive object detection app using YOLOv5/YOLOv8.

🎯 Click "**Start Detection**" to begin real-time object detection.  
Click "**Stop Detection**" to stop it.

Powered by **LUMI MATE AI**.
""")

# ---------------- Main Title ----------------
st.title("🎯 Object Detection")
st.markdown("Click '**Start Detection**' to activate real-time object detection using YOLO. Then click '**Stop Detection**' to stop.")

# Load YOLO model once
model = YOLO("yolov8s.pt")

# Detection processor function for streamlit-webrtc
def process_frame(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    results = model(img)[0]
    annotated_img = results.plot()
    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# Streamlit-webrtc widget
webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode="sendrecv",
    video_processor_factory=lambda: type(
        "YOLOProcessor",
        (),
        {"recv": process_frame}
    )(),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# ---------------- Footer ----------------
st.markdown("""
<hr style="margin-top: 50px; margin-bottom: 10px;">
<div style="text-align: center; font-size: 14px; color: grey;">
    Developed by <strong>LUMI MATE AI</strong>
</div>
""", unsafe_allow_html=True)
