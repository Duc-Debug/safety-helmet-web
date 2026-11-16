import torch
import cv2
import numpy as np
import logging

logger = logging.getLogger("safety-helmet-service")


class YOLOv5Box:
    """Mô phỏng cấu trúc Bounding Box của Ultralytics Results để tương thích pipeline."""
    def __init__(self, x1: float, y1: float, x2: float, y2: float, conf: float, cls_id: int):
        self.xyxy = [torch.tensor([x1, y1, x2, y2])]
        self.conf = [torch.tensor(conf)]
        self.cls = [torch.tensor(cls_id)]


class YOLOv5Results:
    """Mô phỏng đối tượng Ultralytics Results chứa danh sách boxes và names."""
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class YOLOv5Wrapper:
    """Wrapper cho mô hình YOLOv5 checkpoint huấn luyện từ repo gốc (ultralytics/yolov5),
    giúp cung cấp interface `.predict()` và thuộc tính `.names` tương thích hoàn toàn với YOLOv8.
    """
    def __init__(self, model_path: str):
        logger.info(f"Đang nạp mô hình YOLOv5 tùy biến từ {model_path} thông qua Torch Hub...")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, trust_repo=True)
        raw_names = getattr(self.model, 'names', {})
        if isinstance(raw_names, list):
            self.names = {i: n for i, n in enumerate(raw_names)}
        else:
            self.names = raw_names
        logger.info(f"Nạp YOLOv5 thành công! Danh sách lớp: {self.names}")

    def predict(self, source, conf: float = 0.25, iou: float = 0.45, verbose: bool = False):
        """Thực hiện suy luận và trả về danh sách đối tượng giống Ultralytics Results."""
        self.model.conf = conf
        self.model.iou = iou

        # Chuyển đổi định dạng ảnh BGR (OpenCV) sang RGB cho YOLOv5
        if isinstance(source, np.ndarray) and len(source.shape) == 3:
            rgb_img = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = source

        preds = self.model(rgb_img)
        xyxy_tensor = preds.xyxy[0].cpu().numpy()

        boxes = []
        for row in xyxy_tensor:
            x1, y1, x2, y2, c, cls_id = row
            boxes.append(YOLOv5Box(float(x1), float(y1), float(x2), float(y2), float(c), int(cls_id)))

        return [YOLOv5Results(boxes, self.names)]
