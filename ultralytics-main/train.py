
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

model.train(data='seg.yaml',workers=0,epochs=60,batch=16,patience=300)