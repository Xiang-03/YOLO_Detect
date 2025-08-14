from types import SimpleNamespace

import yaml

from ultralytics import YOLO
from ultralytics.data.mosaic_cover import CustomDataset

# 1. 加载模型
model = YOLO(r"E:\ultralytics-main\runs\detect\train12\weights\last.pt")

# 2. 读取数据集配置
with open(r"E:\ultralytics-main\ultralytics\cfg\datasets\magni.yaml", encoding="utf-8") as f:
    data_cfg = yaml.safe_load(f)

# 3. 读取 YOLO 官方完整超参数配置（路径要根据你本地 ultralytics 安装位置调整）
with open(r"E:\ultralytics-main\ultralytics\cfg\default.yaml", encoding="utf-8") as f:
    hyp_dict = yaml.safe_load(f)

# 4. 转成 Namespace，确保 hyp 里所有属性齐全
hyp_namespace = SimpleNamespace(**hyp_dict)

# 5. 创建自定义数据集
custom_data = CustomDataset(
    img_path=r"E:\A_xianghang\data_yolo_copy\YOLODataset\images",
    data=data_cfg,
    imgsz=640,
    augment=True,
    hyp=hyp_namespace,
    rect=False,
    stride=32,
    p=0.5,  # 这里设置 mosaic 概率，根据需要调整
)

if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()  # Windows下可选，但推荐加上

    # 6. 训练模型
    # 训练时
    model.train(
        data=r"E:\ultralytics-main\ultralytics\cfg\datasets\magni.yaml",  # 传路径，不能传对象
        epochs=100,
        imgsz=640,
        batch=16,
        optimizer="AdamW",
        lr0=0.001,
        augment=True,
        resume=True,
    )
