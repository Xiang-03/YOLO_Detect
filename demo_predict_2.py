from ultralytics import YOLO
from ultralytics.data.mosaic_cover import CustomDataset

# 1. 创建模型
model = YOLO("yolov11n.pt")  # 使用你的YOLOv11模型

# 2. 创建自定义数据集
custom_data = CustomDataset(
    data="magni.yaml",
    imgsz=640,
    augment=True,  # 启用增强
    hyp=model.args,  # 使用模型的超参数
    rect=False,     # 确保使用方形图像
    stride=32       # 模型步长
)

# 3. 训练模型
model.train(
    data=custom_data,  # 使用自定义数据集
    epochs=100,
    imgsz=640,
    batch=16,
    optimizer='AdamW',
    lr0=0.001,
    augment=True,  # 确保启用增强
    # 其他训练参数...
)