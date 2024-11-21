import os
import requests
from datetime import datetime
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np

# Load the YOLOv8 model trained for license plate detection
model = YOLO('yolov9s.pt')  # Replace with your model path

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)  # Enable GPU for faster OCR if available

# Path to the video file
video_path = 'sample.mp4'  # Replace with your video path

# Retrieve the database URL from environment variables or use the default
db_url = os.getenv('DATABASE_URL', 'http://10.0.0.36:5000/add_record')

# Open the video file using OpenCV
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Frame processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if no more frames

    # Preprocess frame for YOLO inference
    results = model(frame)

    # Iterate over detected objects
    for result in results:
        for box in result.boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]  # Confidence score for the detection

            if conf < 0.5:  # Skip low-confidence detections
                continue

            # Draw bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

            # Crop the license plate region
            license_plate_image = frame[y1:y2, x1:x2]

            # Convert to grayscale for OCR
            gray_license_plate = cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2GRAY)

            # Perform OCR to extract text
            ocr_result = reader.readtext(gray_license_plate)

            # Extract and display recognized text
            if ocr_result:
                text = ocr_result[0][-2]
                confidence = ocr_result[0][-1]
                confidence = round(confidence, 2)  # Round confidence

                # Format timestamp as a standard date and time string
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Display recognized text on the video
                cv2.putText(frame, f'{text} ({confidence:.2f})', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Log recognized license plate
                print(f'Recognized License Plate: {text}, Confidence: {confidence:.2f}, Timestamp: {timestamp}')

                # Prepare data for database insertion
                data = {
                    'license_plate': text,
                    'confidence': confidence,
                    'timestamp': timestamp
                }

                # Send data to the database
                try:
                    response = requests.post(db_url, json=data)
                    response.raise_for_status()
                    print('Data successfully sent to the database.')
                except requests.exceptions.RequestException as e:
                    print(f'Error sending data to the database: {e}')

    # Display the processed frame
    cv2.imshow('ALPR Video', frame)

    # Press 'q' to quit the video early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
