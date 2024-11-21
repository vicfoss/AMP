import cv2
import threading
from ultralytics import YOLO
from datetime import datetime
import easyocr
import numpy as np

# Load the YOLOv8 model trained for license plate detection
model = YOLO('yolov9s.pt')  # Replace with your model path

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)  # Use GPU if available for faster OCR

# List of RTSP streams or video streams to process
streams = [
    "rtsp://username:password@ip_address1/stream",
    "rtsp://username:password@ip_address2/stream",
    "sample.mp4",  # You can mix RTSP streams and video files
    #"sample2.mp4"
]

# Function to process a single stream
def process_stream(stream_url):
    print(f"[INFO] Processing stream: {stream_url}")
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open stream: {stream_url}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"[INFO] Stream ended: {stream_url}")
            break

        # Perform license plate detection using YOLO
        results = model(frame)

        # Iterate over detected objects
        for result in results:
            for box in result.boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]  # Detection confidence

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

                    # Display recognized text on the video
                    cv2.putText(frame, f'{text} ({confidence:.2f})', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Log recognized license plate
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"[INFO] Stream: {stream_url}, Plate: {text}, Confidence: {confidence}, Time: {timestamp}")

        # Display the frame with bounding boxes and recognized text
        cv2.imshow(f'Stream: {stream_url}', frame)

        # Press 'q' to quit the stream early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"[INFO] Exiting stream: {stream_url}")
            break

    cap.release()
    cv2.destroyWindow(f'Stream: {stream_url}')
    print(f"[INFO] Finished processing stream: {stream_url}")

# Create and start a thread for each stream
threads = []
for stream in streams:
    thread = threading.Thread(target=process_stream, args=(stream,))
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()

print("[INFO] All streams processed.")
