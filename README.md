Pen Detection using YOLO and OpenCV

A simple real-time pen detection program using a custom-trained YOLO model and OpenCV. It detects pens through a webcam and displays bounding boxes with confidence scores.

Features

- Real-time pen detection using a webcam
- Bounding boxes around detected pens
- Confidence score display
- 30-second detection limit
- Press `Q` to exit

Technologies Used

- Python
- OpenCV
- Ultralytics YOLO

How It Works

The program captures frames from the webcam and passes them to the trained YOLO model for inference.

The model returns detected objects, their bounding box coordinates, and confidence scores. Detections with confidence above `0.30` are displayed with bounding boxes and labels.

Installation

Install the required libraries:

pip install opencv-python ultralytics


Running the Program

1. Clone this repository.
2. Make sure your trained YOLO weights (`best.pt`) are available within the same directory.

3) Run the Python script:

python main.py


Future Improvements

- Improve detection accuracy.
- Optimize real-time performance.
- Support multiple object classes.
