from ultralytics import YOLO
from ultralytics.data.mosaic_cover import CustomDataset

# Load a COCO-pretrained YOLOv8n model
model = YOLO("E:\\ultralytics-main\\runs\\detect\\train6\\weights\\best.pt")

# # Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="magni.yaml", epochs=50, imgsz=640,device=[0],workers=0,cache=True,resume=True)


# 2. 创建自定义数据集
custom_data = CustomDataset(
    data="E:\\ultralytics-main\\ultralytics\\cfg\\datasets\\magni.yaml",
    img_path="E:/A_xianghang/data_yolo_copy/YOLODataset/images",
    # labels_path="E:/datasets/your_labels_folder",  # 如果需要
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
)




