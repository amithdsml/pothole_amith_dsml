
## Update this and point to main folder

ROOT_DIR = '/Documents/GitHub/pothole_amith_dsml/pothole_training/'

## call the libraries
import os
from ultralytics import YOLO

## selecting the model
model = YOLO("yolov8n.pt")

## run the training
### set the right yaml file

results = model.train(data = os.path.join(ROOT_DIR, "google_colab_config.yaml"), epochs = 10)
