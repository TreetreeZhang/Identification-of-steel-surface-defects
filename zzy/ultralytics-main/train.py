from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

model.train(data='seg.yaml',workers=0,epochs=60,batch=16,patience=300)