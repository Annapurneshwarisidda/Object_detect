import streamlit as st
import cv2
from ultralytics import YOLO

st.sidebar.title("📄 About This App")
st.sidebar.markdown("""
An interactive object detection app using YOLOv5/YOLOv8.

🎯 Click "**Start Detection**" to begin real-time object detection.  
Click "**Stop Detection**" to stop it.

Powered by **LUMI MATE AI**.
""")

st.title("🎯 Controlled Object Detection")
st.markdown("Click '**Start Detection**' to activate real-time object detection using YOLO. Then click '**Stop Detection**' to stop.")

# Initialize detection state and capture in session_state
if "detecting" not in st.session_state:
    st.session_state.detecting = False
if "cap" not in st.session_state:
    st.session_state.cap = None

col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Detection"):
        st.session_state.detecting = True
        if st.session_state.cap is None:
            st.session_state.cap = cv2.VideoCapture(0)
with col2:
    if st.button("⏹ Stop Detection"):
        st.session_state.detecting = False
        if st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None

model = YOLO("yolov8s.pt")
FRAME_WINDOW = st.image([])

if st.session_state.detecting and st.session_state.cap is not None:
    ret, frame = st.session_state.cap.read()
    if not ret:
        st.error("❌ Failed to read from webcam.")
        st.session_state.detecting = False
        st.session_state.cap.release()
        st.session_state.cap = None
    else:
        results = model(frame, verbose=False)[0]
        annotated_frame = results.plot()
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(rgb_frame)
else:
    FRAME_WINDOW.empty()

st.markdown("""
<hr style="margin-top: 50px; margin-bottom: 10px;">
<div style="text-align: center; font-size: 14px; color: grey;">
    Developed by <strong>LUMI MATE AI</strong>
</div>
""", unsafe_allow_html=True)
