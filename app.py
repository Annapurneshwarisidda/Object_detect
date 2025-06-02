import streamlit as st
import cv2
from ultralytics import YOLO
import speech_recognition as sr
import threading

# ---------------- Sidebar ----------------
st.sidebar.title("📄 About This App")
st.sidebar.markdown("""
A voice-activated object detection app using YOLOv5/YOLOv8.

🎙️ Say "**start detection**" to begin real-time object detection.  
Say "**stop detection**" to end it.

Powered by **LUMI MATE AI**.
""")

# ---------------- Main Title ----------------
st.title("🎙️ Voice-Controlled Object Detection")
st.markdown("Speak '**start detection**' to activate real-time object detection using YOLO. Then say '**stop detection**' to stop.")

FRAME_WINDOW = st.image([])

# Load YOLO model
model = YOLO("yolov8s.pt")

# Shared flag
detecting = False

# Speech recognizer
recognizer = sr.Recognizer()


def listen_for_command(trigger_words):
    """Listen for any phrase in trigger_words list."""
    with sr.Microphone() as source:
        try:
            st.info("🎧 Listening for voice command...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
            response = recognizer.recognize_google(audio).lower()
            if any(word in response for word in trigger_words):
                return response
            return None
        except sr.UnknownValueError:
            st.warning("❗ Could not understand audio.")
            return None
        except sr.RequestError as e:
            st.error(f"⚠️ API Error: {e}")
            return None
        except sr.WaitTimeoutError:
            st.warning("⏱️ Listening timed out.")
            return None
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return None


def voice_listener():
    """Continuously listens for 'stop detection' in background."""
    global detecting
    while detecting:
        command = listen_for_command(["stop detection"])
        if command and "stop detection" in command:
            detecting = False
            break


# Voice command trigger
if st.button("🎤 Listen for Voice Command"):
    command = listen_for_command(["start detection"])
    if command and "start detection" in command:
        st.success("✅ Voice command recognized: Starting detection...")

        detecting = True

        # Start background thread for stop command
        listener_thread = threading.Thread(target=voice_listener)
        listener_thread.start()

        # Start webcam detection
        cap = cv2.VideoCapture(0)
        while detecting:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Could not access webcam.")
                break

            results = model(frame)[0]
            annotated_frame = results.plot()
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(rgb_frame)

        cap.release()
        FRAME_WINDOW.empty()
        st.warning("🛑 Detection stopped by voice command.")
    else:
        st.warning("❗ Say 'start detection' clearly to begin.")

# ---------------- Footer ----------------
st.markdown("""
<hr style="margin-top: 50px; margin-bottom: 10px;">
<div style="text-align: center; font-size: 14px; color: grey;">
    Developed by <strong>LUMI MATE AI</strong>
</div>
""", unsafe_allow_html=True)
