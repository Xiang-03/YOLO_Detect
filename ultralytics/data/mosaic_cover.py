import random

import numpy as np

from ultralytics.data.augment import Mosaic
from ultralytics.data.dataset import YOLODataset


class CustomMosaic(Mosaic):
    """针对 face 和 bicycle 类别的 Mosaic 增强，并添加随机遮挡."""

    def __init__(self, dataset, imgsz=640, p=1.0):
        super().__init__(dataset, imgsz=imgsz, p=p)
        # 获取类别名称到ID的映射
        self.target_class_ids = []
        if hasattr(dataset, "names"):
            for name in ["face", "bicycle"]:
                if name in dataset.names.values():
                    self.target_class_ids.append(dataset.names[name])
        self.occlusion_prob = 0.7  # 遮挡概率
        self.occlusion_count = (1, 3)  # 遮挡次数范围

    def get_indexes(self):
        """筛选包含目标类别的样本索引."""
        indexes = []
        for idx in range(len(self.dataset)):
            label = self.dataset.labels[idx]
            if label is not None and len(label) > 0:
                # 检查标签中是否包含目标类别
                if any(cls in self.target_class_ids for cls in label[:, 0]):
                    indexes.append(idx)
        return indexes if indexes else super().get_indexes()

    def apply_occlusion(self, img, labels):
        """在目标类别的边界框上添加随机遮挡."""
        if not self.target_class_ids or random.random() > self.occlusion_prob:
            return img, labels

        h, w = img.shape[:2]
        # 选择1-3个目标框进行遮挡
        num_occlusions = random.randint(*self.occlusion_count)
        target_boxes = [l for l in labels if l[0] in self.target_class_ids]

        if not target_boxes:
            return img, labels

        # 随机选择要遮挡的框
        boxes_to_occlude = random.sample(target_boxes, min(num_occlusions, len(target_boxes)))

        for label in boxes_to_occlude:
            # 转换为像素坐标
            cls, x_center, y_center, box_w, box_h = label[:5]
        x1 = int((x_center - box_w / 2) * w)
        y1 = int((y_center - box_h / 2) * h)
        x2 = int((x_center + box_w / 2) * w)
        y2 = int((y_center + box_h / 2) * h)

        # 确保坐标在图像范围内
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        if x2 > x1 and y2 > y1:
            # 创建不同类型的遮挡
            occlusion_type = random.choice(["noise", "solid", "grid"])

            if occlusion_type == "noise":
                # 随机噪声遮挡
                patch = np.random.randint(0, 256, (y2 - y1, x2 - x1, 3), dtype=np.uint8)
                img[y1:y2, x1:x2] = patch

            elif occlusion_type == "solid":
                # 纯色遮挡
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                img[y1:y2, x1:x2] = color

            elif occlusion_type == "grid":
                # 网格遮挡
                for i in range(y1, y2, 10):
                    for j in range(x1, x2, 10):
                        if i < y2 and j < x2:
                            img[i : i + 5, j : j + 5] = (0, 0, 0)

        return img, labels

    def __call__(self, labels):
        """重载 Mosaic 逻辑，添加遮挡."""
        if self.dataset.p == 0.0:
            return labels

        # 调用原始 Mosaic 增强
        img, labels = super().__call__(labels)

        # 应用自定义遮挡
        return self.apply_occlusion(img, labels)


class CustomDataset(YOLODataset):
    """自定义数据集，使用针对特定类别的增强."""

    def __init__(self, *args, p=1.0, **kwargs):
        # 确保 data 参数是字典而不是字符串
        if "data" in kwargs and isinstance(kwargs["data"], str):
            data_path = kwargs["data"]
            import yaml

            with open(data_path, encoding="utf-8") as f:
                kwargs["data"] = yaml.safe_load(f)

        super().__init__(*args, **kwargs)

        self.p = p  # mosaic概率赋值，默认1.0

        # 替换默认的 Mosaic 增强
        self.mosaic = CustomMosaic(self, self.imgsz, self.p)

    def build_transforms(self, hyp=None):
        """构建数据增强管道."""
        transforms = super().build_transforms(hyp)
        # 找到 Mosaic 的位置并替换
        for i, t in enumerate(transforms.transforms):
            if isinstance(t, Mosaic):
                transforms.transforms[i] = self.mosaic
                break
        return transforms
