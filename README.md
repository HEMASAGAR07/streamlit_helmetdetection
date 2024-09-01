# streamlit_helmetdetection
# Helmet and Number Plate Detection

This project focuses on detecting motorcycle riders with or without helmets and recognizing number plates of vehicles using object detection and Optical Character Recognition (OCR) techniques.

## Overview

The system detects the following key components:
- **Riders with helmets**
- **Riders without helmets**
- **Number plates**

The objective is to ensure compliance with helmet safety regulations by detecting riders who do not wear helmets and to recognize their vehicle's number plate for further processing.

## Technologies Used

1. **YOLO (You Only Look Once)**: 
   - **Version**: YOLOv8
   - **Use**: Object detection for identifying riders, helmets, and number plates. YOLO is a fast and accurate object detection algorithm that processes the entire image at once, making it suitable for real-time applications.

2. **OpenCV**:
   - **Use**: Image processing and computer vision tasks such as frame extraction, drawing bounding boxes, and handling video streams.

3. **PaddleOCR**:
   - **Use**: Optical Character Recognition (OCR) to extract text from number plates. It’s efficient and supports various languages. The OCR extracts alphanumeric text and provides confidence scores.

4. **Streamlit**:
   - **Use**: Front-end framework to create an interactive web application for uploading and processing images and videos. It allows users to view the results directly in the browser.

5. **PyTorch**:
   - **Use**: For model loading and inference, ensuring that the YOLO model can leverage GPU processing if available.

6. **cvzone**:
   - **Use**: Simplified OpenCV functions for drawing rectangles and text on images.

## How It Works

1. **Object Detection**:
   - The YOLO model processes each frame from the input (image or video) to detect objects like riders, helmets, and number plates.
   - Bounding boxes are drawn around the detected objects with a label indicating the class (e.g., "Rider", "Helmet", "No Helmet", "Number Plate").

2. **Number Plate Recognition**:
   - When a number plate is detected, the relevant section of the image is cropped.
   - The cropped image is passed through PaddleOCR to extract and recognize the alphanumeric characters from the plate.
   - If the confidence score of the recognized text is above a threshold, the number is displayed on the image.

3. **Real-Time Processing**:
   - The system can process video files frame by frame and provide real-time feedback on the detected objects and recognized text.

## Challenges Faced

- **Accuracy in Detection**: Achieving high accuracy in detecting riders and helmets required tuning the YOLO model, adjusting confidence thresholds, and experimenting with different versions of YOLO.
- **OCR Errors**: OCR systems can sometimes misinterpret text, especially if the number plate is unclear or distorted. Improving recognition accuracy involved experimenting with different OCR libraries and tuning parameters like confidence thresholds.
- **Integration with Streamlit**: Displaying the results in a user-friendly way required careful integration between the backend (model processing) and the front-end Streamlit app.
- **Handling Large Video Files**: Processing large video files in real-time can be resource-intensive, so optimizing the frame processing loop and managing memory efficiently was crucial.

## Setup and Installation

To run the project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/helmet-number-plate-detection.git
   cd helmet-number-plate-detection
# Helmet and Number Plate Detection

## Overview
This project aims to detect riders without helmets and recognize number plates using YOLO (You Only Look Once) for object detection and OCR (Optical Character Recognition) for number plate recognition. The system is built to help enforce traffic rules by automatically detecting violations and logging the relevant information.

## Features
- **Helmet Detection**: Detects whether a rider is wearing a helmet or not.
- **Number Plate Detection**: Recognizes and reads number plates of vehicles.
- **Object Detection**: Detects riders, helmets, and number plates in real-time using YOLOv8.
- **OCR Integration**: Recognizes number plate text using PaddleOCR (or any OCR tool).

## Technologies Used
- **YOLOv8**: For real-time object detection, particularly for detecting riders, helmets, and number plates.
- **PaddleOCR**: For extracting text from detected number plates.
- **OpenCV**: For image and video processing.
- **Streamlit**: For deploying the model as a web application, providing an interactive interface to upload images or videos and visualize detections.
- **Python Libraries**: Various Python libraries including NumPy, cvzone, and PIL for image manipulation and mathematical operations.
  
## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/helmet-number-plate-detection.git
    cd helmet-number-plate-detection
    ```

2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit application**:
    ```bash
    streamlit run main.py
    ```

## How It Works
- **YOLOv8** is used for detecting multiple objects in the image or video frames. It identifies the rider, helmet, and number plate.
- The detected number plate is passed to **PaddleOCR** (or any OCR tool) to recognize the alphanumeric characters on the plate.
- The processed frames or images are displayed using Streamlit, with bounding boxes and labels indicating the detected objects.

  
## Future Enhancements
- **Multiple Languages Support**: Expanding the OCR capabilities to support number plates in different languages.
- **Night Mode Detection**: Enhancing the system to work efficiently under low-light conditions using infrared cameras or improved image preprocessing.
- **Scalability**: Deploying the system on cloud platforms with real-time alert systems integrated into traffic monitoring infrastructure.
- **Vehicle Classification**: Adding features to classify vehicles (e.g., cars, bikes) and logging the type of vehicle along with the detected helmet status and number plate.

## Acknowledgments
We would like to express our sincere gratitude to the following:
- **YOLOv8 Community**: For providing a powerful object detection framework.
- **PaddleOCR Developers**: For developing an easy-to-use OCR tool that enabled accurate number plate recognition.
- **OpenCV Contributors**: For providing robust tools for image and video processing.
- **Streamlit**: For creating a flexible framework to deploy machine learning models as web applications.


