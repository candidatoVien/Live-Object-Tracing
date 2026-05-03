import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
import numpy as np


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()


st.sidebar.title("ℹ️ Project Information")
st.sidebar.info(
    "This application is a real-time AI tool for 'Activity 03'. "
    "It detects and tracks multiple objects simultaneously using YOLOv8."
)

st.sidebar.subheader("🚀 Step-by-Step Guide")
st.sidebar.markdown(
    """
    1. **Allow Permissions**: Click 'Allow' for camera access.
    2. **Start Stream**: Click the **Start** button to begin.
    3. **Detect Objects**: Point your camera at people, phones, or bottles.
    4. **View Alerts**: A red border appears if a Person or Phone is detected.
    5. **Custom Style**: Use the color picker below to change the background!
    """
)


bg_color = st.sidebar.color_picker("Pick a background color", "#800000")


st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {bg_color};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# 4. Main Page UI
st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

def video_frame_callback(frame):
    # Convert WebRTC frame to BGR for processing
    img = frame.to_ndarray(format="bgr24")

    
    results = model.track(img, persist=True, conf=0.35, imgsz=640, verbose=False)

    
    annotated_frame = results[0].plot()

    
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    
    if results[0].boxes is not None:
        names = model.names
        detected_classes = results[0].boxes.cls.cpu().numpy()
        
        total_count = len(detected_classes)
        counts = {}
        for cls_idx in detected_classes:
            label = names[int(cls_idx)].capitalize()
            counts[label] = counts.get(label, 0) + 1

        # ALERT TRIGGER: Visual red border if specific items are found
        if "Person" in counts or "Cell phone" in counts:
            cv2.rectangle(annotated_frame, (0, 0), (img.shape[1], img.shape[0]), (255, 0, 0), 15)

        
        y_offset = 40
        font_scale = 0.7
        thickness = 2

       
        total_text = f"Object (total): {total_count}"
        cv2.putText(annotated_frame, total_text, (22, y_offset + 2), 
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(annotated_frame, total_text, (20, y_offset), 
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 0), thickness)
        
        y_offset += 35

       
        for label, count in counts.items():
            item_text = f"{label}: {count}"
            cv2.putText(annotated_frame, item_text, (22, y_offset + 2), 
                        cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), thickness + 1)
            cv2.putText(annotated_frame, item_text, (20, y_offset), 
                        cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 0), thickness)
            y_offset += 30 

    return av.VideoFrame.from_ndarray(annotated_frame, format="rgb24")


webrtc_streamer(
    key="precision-counter-app",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640, "min": 640}, 
            "height": {"ideal": 480, "min": 480},
        },
        "audio": False
    },
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
)