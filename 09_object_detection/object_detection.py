import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open CCTV/video file
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Detect objects
    results = model(frame)

    # Draw detections on the frame
    annotated_frame = results[0].plot()

    # Display result
    cv2.imshow("CCTV Object Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()