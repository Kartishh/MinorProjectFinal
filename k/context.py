import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np
from PIL import Image

st.set_page_config(page_title="Context-Aware Object Detection", layout="wide")

st.title("🧠 Context-Aware Object Detection (YOLO + Streamlit)")
st.markdown("This app uses your webcam and a YOLO model to detect objects in real-time.")

# Load YOLO model
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")  # change to your trained model path if needed
    return model

model = load_model()

# Start camera
run_camera = st.checkbox("Start Camera", value=False)
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run_camera:
    ret, frame = camera.read()
    if not ret:
        st.warning("Failed to access webcam.")
        break

    # Convert frame for YOLO
    results = model.predict(source=frame, conf=0.4, verbose=False)

    # Draw results on frame
    annotated_frame = results[0].plot()

    FRAME_WINDOW.image(annotated_frame, channels="BGR")

camera.release()
