import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image

# ---------------- Sidebar ----------------
st.sidebar.title("📄 About This App")
st.sidebar.markdown("""
An interactive object detection app using YOLOv5/YOLOv8.

🎯 Click "**Start Detection**" to begin real-time object detection.  
Click "**Stop Detection**" to stop it.

Powered by **LUMI MATE AI**.
""")

# ---------------- Main Title ----------------
st.title("🎯 Button-Controlled Object Detection")
st.markdown("Click '**Start Detection**' to activate real-time object detection using YOLO. Then click '**Stop Detection**' to stop.")

# Frame container
FRAME_WINDOW = st.image([])

# Load YOLO model
model = YOLO("yolov8s.pt")

# ---------------- Detection Control ----------------
# Use session state to persist detection state
if "detecting" not in st.session_state:
    st.session_state.detecting = False

col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Detection"):
        st.session_state.detecting = True
with col2:
    if st.button("⏹ Stop Detection"):
        st.session_state.detecting = False

# ---------------- Detection Loop ----------------
if st.session_state.detecting:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Could not access the webcam.")
    else:
        st.success("✅ Detection started. Click 'Stop Detection' to end.")
        while st.session_state.detecting:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to read from webcam.")
                break

            results = model(frame, verbose=False)[0]
            annotated_frame = results.plot()
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(rgb_frame)

        cap.release()
        FRAME_WINDOW.empty()
        st.info("🛑 Detection stopped.")

# ---------------- Footer ----------------
st.markdown("""
<hr style="margin-top: 50px; margin-bottom: 10px;">
<div style="text-align: center; font-size: 14px; color: grey;">
    Developed by <strong>LUMI MATE AI</strong>
</div>
""", unsafe_allow_html=True)
