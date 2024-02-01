from ultralytics import YOLO
import torch

import os

os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
import sys

sys.argv.append("-Xfrozen_modules=off")

device = torch.device("cpu")

# Load a model
model = YOLO("yolov8s.yaml").to(
    device
)  # Make sure to move the model to the desired device

# Use the model
# Assuming your YOLO class has a train method
result = model.train(data="Code/config.yaml", epochs=100, device=device)
