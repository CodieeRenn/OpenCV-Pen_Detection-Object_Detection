import cv2
import time
from ultralytics import YOLO

# Load model

# make sure to correct the path
model = YOLO('./runs/detect/train/weights/best.pt')

# Open webcam
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("YOLO Real-Time Detection", cv2.WINDOW_NORMAL)

start_time = time.time()

try:
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = result[:6]
            if conf > 0.3:
                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("YOLO Real-Time Detection", frame)
        cv2.waitKey(1)

        # Stop after 30 seconds
        if time.time() - start_time > 30:
            print("Time limit reached.")
            break

        # Or exit if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    video_capture.release()
    cv2.destroyAllWindows()
