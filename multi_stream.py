import os
import cv2
import threading
from ultralytics import YOLO
from datetime import datetime
import easyocr
import requests
import numpy as np

# Load the YOLOv8 model trained for license plate detection (lightweight version)
model = YOLO('yolov9s.pt')  # Replace with your model path

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)  # Use GPU if available for faster OCR

# List of NTSC camera streams
streams = [
    "sample.mp4"
    # "rtsp://username:password@ip_address1/stream",
    # "rtsp://username:password@ip_address2/stream",
    # Add remaining 6 camera streams here
]

# Get the Flask API URL from the environment variable, with a default fallback
API_URL = os.getenv('DATABASE_URL', 'http://10.0.0.36:5000/add_record')

# Function to send data to the Flask API
def save_to_remote(plate, timestamp, confidence):
    try:
        data = {"plate": plate, "timestamp": timestamp, "confidence": confidence}
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            print(f"[INFO] Data sent to remote database: Plate={plate}, Timestamp={timestamp}, Confidence={confidence}")
        else:
            print(f"[ERROR] Failed to send data: {response.text}")
    except Exception as e:
        print(f"[ERROR] Failed to connect to remote database: {e}")

# Function to process a single stream
def process_stream(stream_url, stream_id):
    print(f"[INFO] Processing stream: {stream_url}")
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print(f"[ERROR] Could not open stream: {stream_url}")
        return

    frame_count = 0
    FRAME_SKIP = 5  # Process one frame every 5 frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"[INFO] Stream ended: {stream_url}")
            break

        # Downscale input frames for faster processing
        frame = cv2.resize(frame, (640, 360))  # Resize to lower resolution

        # Skip frames to limit FPS processing
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        # Perform license plate detection using YOLO
        results = model(frame)

        # Iterate over detected objects
        for result in results:
            for box in result.boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]  # Detection confidence

                # Draw bounding box on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

                # Crop the license plate region
                license_plate_image = frame[y1:y2, x1:x2]

                # Convert to grayscale for OCR
                gray_license_plate = cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2GRAY)

                # Perform OCR to extract text
                ocr_result = reader.readtext(gray_license_plate)

                if ocr_result:
                    text = ocr_result[0][-2]
                    confidence = ocr_result[0][-1]
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    # Print recognized license plate information
                    print(f"[INFO] Plate: {text}, Confidence: {confidence}, Timestamp: {timestamp}")

                    # Send all recognized plates to the remote database
                    save_to_remote(text, timestamp, confidence)

                    # Display recognized text on the video
                    cv2.putText(frame, f'{text} ({confidence:.2f})', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the processed frame
        cv2.imshow(f'Stream {stream_id}', frame)

        # Press 'q' to quit the video early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"[INFO] Exiting stream: {stream_url}")
            break

    # Release resources
    cap.release()
    cv2.destroyWindow(f'Stream {stream_id}')
    print(f"[INFO] Finished processing stream: {stream_url}")

# Create and start a thread for each stream
threads = []
for idx, stream in enumerate(streams):
    thread = threading.Thread(target=process_stream, args=(stream, idx))
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()

print("[INFO] All streams processed.")
