import os
import cv2
import streamlit as st
import numpy as np
from ultralytics import YOLO

# Define folders
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
CROPPED_FOLDER = "static/cropped"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FOLDER, exist_ok=True)

# Load YOLO models
person_model = YOLO("weights/person_detection.pt")
ppe_model = YOLO("weights/ppe_detection.pt")

# Define PPE class names
PPE_CLASSES = ["hard-hat", "gloves", "mask", "glasses", "boots", "vest", "ppe-suit", "ear-protector", "safety-harness"]

# Streamlit page config
st.set_page_config(page_title="PPE Detection", layout="wide")

# Custom CSS for dark mode UI
st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: #ffffff;
    }
    .title {
        color: #1abc9c; 
        text-align: center;
        font-size: 36px;
        font-weight: bold;
    }
    .subheader {
        color: #bdc3c7;
        text-align: center;
        font-size: 18px;
        background-color: rgba(0, 0, 0, 0.7); 
        padding: 8px;
        border-radius: 10px;
        display: inline-block;
    }
    .upload-box {
        border: 2px dashed #1abc9c;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        background-color: #1e272e;
        font-weight: bold;
    }
    .download-btn {
        background-color: #3498db !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 8px !important;
    }
    .sidebar {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
    }
    .footer {
        position: fixed;
        bottom: 10px;
        left: 50%;
        transform: translateX(-50%);
        text-align: center;
        color: #888888;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.markdown("<h2 style='color:#1abc9c;'>⚙ Settings</h2>", unsafe_allow_html=True)
theme_option = st.sidebar.radio("Choose Theme", ["Dark Mode", "Light Mode"], index=0)
st.sidebar.markdown("---")

# Apply theme
if theme_option == "Light Mode":
    st.markdown(
        """
        <style>
        body {background-color: #f4f4f4; color: #000;}
        .title {color: #2c3e50;}
        .subheader {color: #34495e; background-color: rgba(255, 255, 255, 0.7);}
        .upload-box {border: 2px dashed #3498db; background-color: #eaf2ff;}
        .sidebar {background-color: #ffffff;}
        </style>
        """,
        unsafe_allow_html=True
    )

# Page Title
st.markdown("<h1 class='title'>🔍 PPE Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>📤 Upload an image to detect persons and PPE equipment.</p>", unsafe_allow_html=True)

# File Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

def draw_text_with_bg(img, text, x, y, font=cv2.FONT_HERSHEY_SIMPLEX, 
                      font_scale=0.7, font_thickness=2, text_color=(0, 255, 0), bg_color=(0, 0, 0)):
    """Draws text with a background for better visibility."""
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    x_end, y_end = x + text_size[0] + 10, y - text_size[1] - 5

    cv2.rectangle(img, (x, y - text_size[1] - 5), (x_end, y_end + 10), bg_color, -1)
    cv2.putText(img, text, (x + 5, y - 5), font, font_scale, text_color, font_thickness)

def detect_ppe(image_path):
    """Performs person and PPE detection, returning the processed image path."""
    image = cv2.imread(image_path)
    original_image = image.copy()

    # Step 1: Detect Persons
    person_results = person_model(image)
    cropped_persons = []

    for result in person_results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(original_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            draw_text_with_bg(original_image, "Person", x1, y1, text_color=(255, 0, 0))

            cropped_image = image[y1:y2, x1:x2].copy()
            cropped_path = os.path.join(CROPPED_FOLDER, f"person_{idx}.jpg")
            cv2.imwrite(cropped_path, cropped_image)
            cropped_persons.append((cropped_image, (x1, y1)))

    # Step 2: Detect PPE on Cropped Persons
    for cropped_image, (x1, y1) in cropped_persons:
        ppe_results = ppe_model(cropped_image)

        for ppe_result in ppe_results:
            ppe_boxes = ppe_result.boxes.xyxy.cpu().numpy()
            ppe_classes = ppe_result.boxes.cls.cpu().numpy()

            for i, ppe_box in enumerate(ppe_boxes):
                px1, py1, px2, py2 = map(int, ppe_box)
                px1 += x1
                py1 += y1
                px2 += x1
                py2 += y1

                class_id = int(ppe_classes[i])
                label = PPE_CLASSES[class_id] if class_id < len(PPE_CLASSES) else "PPE"

                cv2.rectangle(original_image, (px1, py1), (px2, py2), (0, 255, 0), 2)
                draw_text_with_bg(original_image, label, px1, py1, text_color=(0, 255, 0))

    # Save the output image
    result_path = os.path.join(RESULT_FOLDER, os.path.basename(image_path))
    cv2.imwrite(result_path, original_image)

    return result_path

if uploaded_file:
    # Save uploaded image
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show uploaded image
    st.image(file_path, caption="Uploaded Image", use_column_width=True)

    # Perform inference
    st.markdown("### Processing Image... ⏳")
    result_image_path = detect_ppe(file_path)

    # Show result image
    st.image(result_image_path, caption="Detected PPE", use_column_width=True)

    # Download button for processed image
    with open(result_image_path, "rb") as file:
        st.download_button(
            label="📥 Download Processed Image",
            data=file,
            file_name="ppe_detected.jpg",
            mime="image/jpeg",
            help="Click to download the processed image",
        )

# Footer
st.markdown("<p class='footer'>Developed by <b>Taufiq Ahmed I</b></p>", unsafe_allow_html=True)
