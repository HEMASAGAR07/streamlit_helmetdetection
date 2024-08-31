import streamlit as st
import cv2
import math
import cvzone
import numpy as np
from PIL import Image
from ultralytics import YOLO

from paddleocr import PaddleOCR
import os
import tempfile
import re

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load YOLO model
model = YOLO("best (3).pt")  # Update the path to your best.pt file


classNames = ["number plate", "rider", "with helmet", "without helmet"]
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Initialize OCR


def log_detection(class_name, x1, y1, x2, y2, conf):
    print(f"Detected {class_name} at [{x1}, {y1}, {x2}, {y2}] with confidence {conf}")
def predict_number_plate(img, ocr):
    result = ocr.ocr(img, cls=True)
    result = result[0]
    texts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    if (scores[0]*100) >= 90:
        return re.sub(r'[^a-zA-Z0-9]', '', texts[0]), scores[0]
    else:
        return None, None

def process_frame(img):
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(new_img, stream=True, device=device)

    rider_box = []

    for r in results:
        boxes = r.boxes
        xy = boxes.xyxy
        confidences = boxes.conf
        classes = boxes.cls
        new_boxes = torch.cat((xy.to(device), confidences.unsqueeze(1).to(device), classes.unsqueeze(1).to(device)), 1)
        new_boxes = new_boxes[new_boxes[:, 4].sort()[1]]

        for box in new_boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box[4] * 100)) / 100
            cls = int(box[5])

            log_detection(classNames[cls], x1, y1, x2, y2, conf)

            if classNames[cls] == "without helmet" and conf >= 0.1:
                color = (0, 0, 255)  # Red for without helmet
                cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=5, colorR=color)
                cvzone.putTextRect(img, f"NO HELMET {conf}", (x1 + 10, y1 - 10), scale=1.5,
                                   offset=10, thickness=2, colorT=(255, 255, 255), colorR=color)

            elif classNames[cls] == "with helmet" and conf >= 0.1:
                color = (0, 255, 0)  # Green for helmet
                cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=5, colorR=color)
                cvzone.putTextRect(img, f"HELMET {conf}", (x1 + 10, y1 - 10), scale=1.5,
                                   offset=10, thickness=2, colorT=(255, 255, 255), colorR=color)

            elif classNames[cls] == "rider" and conf >= 0.1:
                color = (255, 0, 0)  # Blue for rider
                cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=5, colorR=color)
                cvzone.putTextRect(img, f"RIDER {conf}", (x1 + 10, y1 - 10), scale=1.5,
                                   offset=10, thickness=2, colorT=(255, 255, 255), colorR=color)

                rider_box.append((x1, y1, x2, y2))

            elif classNames[cls] == "number plate" and conf >= 0.1:
                color = (255, 255, 0)  # Yellow for number plate
                cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=5, colorR=color)
                cvzone.putTextRect(img, f"PLATE {conf}", (x1 + 10, y1 - 10), scale=1.5,
                                   offset=10, thickness=2, colorT=(255, 255, 255), colorR=color)

                crop = img[y1:y2, x1:x2]
                try:
                    vehicle_number, conf_text = predict_number_plate(crop, ocr)
                    if vehicle_number and conf_text:
                        cvzone.putTextRect(img, f"{vehicle_number} {round(conf_text * 100, 2)}%",
                                           (x1, y1 - 50), scale=1.5, offset=10,
                                           thickness=2, colorT=(39, 40, 41),
                                           colorR=(105, 255, 255))
                except Exception as e:
                    print(f"OCR Error: {e}")

    return img


def process_and_display_video(file):
    cap = cv2.VideoCapture(file.name)
    if not cap.isOpened():
        st.error("Error opening video file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create a buffer for the video
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_file.close()

    # Define codec and create VideoWriter object to save the output video
    out = cv2.VideoWriter(temp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Streamlit video display placeholder
    video_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)  # Process each frame

        # Convert frame to RGB and then to PIL Image
        pil_img = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))

        # Display the processed frame
        video_placeholder.image(pil_img, use_column_width=True)

    cap.release()
    out.release()

    # Return the path to the processed video
    return temp_file.name


def process_image(image_file):
    file_bytes = image_file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is not None:
        processed_img = process_frame(img)
        return processed_img
    else:
        st.error("Error decoding image file.")
        return None


# Streamlit interface
st.title("YOLO Object Detection with Streamlit")

uploaded_file = st.file_uploader("Choose an image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    st.write("Processing...")

    if uploaded_file.type.startswith('video'):
        # For video files
        process_and_display_video(uploaded_file)
    else:
        # For image files
        processed_img = process_image(uploaded_file)
        if processed_img is not None:
            st.image(processed_img, channels="BGR", use_column_width=True)
