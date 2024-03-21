from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = '/Users/amithmg/Documents/GitHub/project_video/static'
DETECTION_FOLDER = '/Users/amithmg/Documents/GitHub/project_video'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(DETECTION_FOLDER):
    os.makedirs(DETECTION_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTION_FOLDER'] = DETECTION_FOLDER

# Load YOLO model and classes
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def detect_objects(image):
    height, width, channels = image.shape

    # Detect objects
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Draw bounding box and label
                cv2.rectangle(image, (center_x - w // 2, center_y - h // 2), (center_x + w // 2, center_y + h // 2), (0, 255, 0), 2)
                cv2.putText(image, classes[class_id], (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return redirect(request.url)
    file = request.files['video']
    if file.filename == '':
        return redirect(request.url)
    if file and file.filename.endswith('.mp4'):
        filename = file.filename
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        return redirect(url_for('detect_objects', video_filename=filename))
    else:
        return 'Error: Please upload a valid MP4 video file.'

@app.route('/detect_objects/<video_filename>')
def detect_objects(video_filename):
    cap = cv2.VideoCapture(os.path.join(app.config['UPLOAD_FOLDER'], video_filename))

    if not cap.isOpened():
        return 'Error: Unable to open video file.'

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(os.path.join(app.config['DETECTION_FOLDER'], 'output.avi'), cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply object detection
        detected_frame = detect_objects(frame)

        # Write frame with detections to output video
        out.write(detected_frame)

    cap.release()
    out.release()

    return send_from_directory(app.config['DETECTION_FOLDER'], 'output.avi', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
