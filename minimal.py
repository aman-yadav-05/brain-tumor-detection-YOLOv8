from ultralytics import YOLO

# Try loading the YOLO model
model = YOLO("yolov8n.pt")

# Try training it for 1 epoch just to test
model.train(
    data="data.yaml",
    epochs=1,
    imgsz=640
)
