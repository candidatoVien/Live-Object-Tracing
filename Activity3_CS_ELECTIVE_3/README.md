🎥 Live Object Detection & Tracing

A high-performance, real-time computer vision web application built with Streamlit and YOLOv8. This project provides object detection, tracking, and localized counting with a fully customizable user interface.

✨ Key Features

Precision Multi-Object Tracking: Utilizes the YOLOv8 model with identity persistence (persist=True) to ensure smooth, continuous tracking of moving objects across frames.

Hierarchical Itemized Counter: Displays a "Master Counter" for total objects and a specific category breakdown (e.g., Spoon: 1, Cup: 2, Person: 1) in high-contrast yellow typography.

Visual Alert System: Triggers an immediate full-frame red border alert when target objects (such as People or Cell phones) are detected.

HD Stream Quality: Hard-coded 720p HD resolution constraints (1280x720) to ensure clear detection of smaller items like utensils and tools.

Dynamic UI Customization: Features a default maroon background that can be changed in real-time via a sidebar color picker to suit user preference.

🛠️ Tech Stack

Streamlit: Web interface and interactive UI components.

Ultralytics YOLOv8: Real-time object detection and tracking engine.

OpenCV (cv2): Image processing, BGR-to-RGB conversion, and custom UI overlays.

Streamlit-webrtc: Low-latency browser-based video streaming.

PyAV: Video frame management and array conversion.

🚀 How to Use

Select Background: Use the color picker in the sidebar to set your preferred workspace color (Maroon is the default).

Allow Permissions: Grant the browser access to your webcam when prompted.

Start the Stream: Click the "Start" button to initialize the AI model and video feed.

Detect & Count: Point your camera at objects. The AI will instantly draw bounding boxes and update the itemized list in the top-left corner.

📂 Activity 03 Requirements Met

This application fully satisfies the following requirements from the activity guidelines:

Live Camera Detection: Instant recognition and labeling of everyday objects.

Smooth Tracking: Continuous tracking behavior where objects keep their identity across frames.

Observation Proof: An integrated counting system designed to facilitate the capture of the 5 different object output screenshots required for the formal report.

📝 Setup & Installation

Bash
# Clone the repository
git clone https://github.com/[your-username]/activity-03-ai-counter.git

# Install required dependencies
pip install streamlit streamlit-webrtc ultralytics opencv-python av numpy

# Launch the application
streamlit run app.py