from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("E:\\ultralytics-main\\runs\\detect\\train6\\weights\\best.pt")

# # Define remote image or video URL
# source = "E:\\ultralytics-main\\car.mp4"
#
# # Run inference on the source
# results = model(source)  # list of Results objects


# Run inference on 'bus.jpg' with arguments
model.predict("E:\\ultralytics-main\\ultralytics\\assets\\car.mp4", save=True, imgsz=320, conf=0.3)
