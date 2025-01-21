from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO
import random
import time

# Initialize Flask app
app = Flask(__name__)

# Load YOLO model
yolo = YOLO('yolov9s.pt')

# Video capture from webcam
videoCap = cv2.VideoCapture(0)
videoCap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
videoCap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.4

# Generate a consistent color palette
random.seed(42)
COLORS = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(80)]


def getColours(cls_num):
    return COLORS[cls_num % len(COLORS)]


def generate_frames():
    """Yield frames for the video stream."""
    while True:
        ret, frame = videoCap.read()
        if not ret:
            break

        # Run YOLO object detection
        results = yolo.track(frame, stream=True)

        for result in results:
            classes_names = result.names

            for box in result.boxes:
                if box.conf[0] > CONFIDENCE_THRESHOLD:
                    # Extract bounding box coordinates
                    [x1, y1, x2, y2] = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Get the class and confidence
                    cls = int(box.cls[0])
                    class_name = classes_names[cls]
                    confidence = box.conf[0]

                    # Get respective color
                    colour = getColours(cls)

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(frame, f'{class_name} {confidence:.2f}',
                                (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

        # Encode the frame to JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Provide the video stream to the browser."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True,port=5000)
