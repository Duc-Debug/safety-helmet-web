import os
import io
import base64
import logging
from typing import List, Dict, Any
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

from engine.schemas import (
    DetectedObject,
    WorkerEvaluation,
    SummaryMetrics,
    AnalysisResponse,
)
from engine.rule_engine import SafetyRuleEngine, CLASS_NAMES

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("safety-helmet-service")

# Đường dẫn model và thư mục
WEIGHTS_PATH = os.path.join("weights", "best.pt")
FALLBACK_MODEL_NAME = "yolov8n.pt"
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Khởi tạo instance và state
model: YOLO = None
rule_engine = SafetyRuleEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời ứng dụng FastAPI: tải trọng số mô hình khi khởi động."""
    global model
    if os.path.exists(WEIGHTS_PATH):
        logger.info(f"Đang tải trọng số chuyên dụng từ: {WEIGHTS_PATH}")
        try:
            model = YOLO(WEIGHTS_PATH)
            logger.info("Tải trọng số best.pt thành công!")
        except Exception as e:
            logger.error(f"Lỗi khi tải {WEIGHTS_PATH}: {e}. Chuyển sang fallback model.")
            model = YOLO(FALLBACK_MODEL_NAME)
    else:
        logger.warning(
            f"CẢNH BÁO: Không tìm thấy file {WEIGHTS_PATH}! "
            f"Hệ thống tự động fallback sang mô hình tiêu chuẩn '{FALLBACK_MODEL_NAME}'."
        )
        model = YOLO(FALLBACK_MODEL_NAME)
    yield
    logger.info("Ứng dụng đang dừng hoạt động...")


app = FastAPI(
    title="Safety Helmet Detection & Rule-Based Reasoning API",
    description="Hệ thống nhận diện mũ bảo hộ và trang bị PPE kết hợp lập luận suy diễn luật",
    version="1.0.0",
    lifespan=lifespan
)

# Bật CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đảm bảo thư mục static tồn tại trước khi mount
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", summary="Trang chủ giao diện người dùng")
async def serve_index():
    """Phục vụ file giao diện web chính static/index.html (Chế độ tiêu chuẩn đã khóa tinh chỉnh)."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_file):
        raise HTTPException(status_code=404, detail="File static/index.html chưa được tạo.")
    return FileResponse(index_file)


@app.get("/tuning", summary="Trang giao diện tinh chỉnh tham số chuyên sâu")
async def serve_tuning():
    """Phục vụ file giao diện tinh chỉnh riêng biệt static/tuning.html cho kỹ sư và chuyên gia."""
    tuning_file = os.path.join(STATIC_DIR, "tuning.html")
    if not os.path.exists(tuning_file):
        raise HTTPException(status_code=404, detail="File static/tuning.html chưa được tạo.")
    return FileResponse(tuning_file)


def draw_visualizations(
    image: np.ndarray,
    detections: List[DetectedObject],
    evaluations: List[WorkerEvaluation]
) -> np.ndarray:
    """Vẽ bounding boxes của PPE và công nhân kèm nhãn suy luận lên ảnh."""
    annotated = image.copy()
    img_h, img_w = annotated.shape[:2]

    # Bảng màu cho trạng thái công nhân (BGR)
    STATUS_COLORS: Dict[str, tuple] = {
        "COMPLIANT": (0, 200, 0),             # Xanh lá
        "WARNING_NO_VEST": (0, 165, 255),      # Cam
        "DANGER_NO_HELMET": (0, 0, 255),       # Đỏ
        "INSUFFICIENT_EVIDENCE": (160, 160, 160)  # Xám
    }

    # 1. Vẽ các vật thể PPE nhỏ (helmet, vest, boots, gloves, goggles) bằng viền mảnh màu vàng nhạt
    ppe_color = (0, 220, 255)  # Vàng chanh BGR
    for det in detections:
        if det.cls_name != "person":
            bx1, by1, bx2, by2 = [int(v) for v in det.bbox]
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), ppe_color, 1, cv2.LINE_AA)
            label = f"{det.cls_name} {det.conf:.2f}"
            t_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(
                annotated,
                (bx1, max(0, by1 - t_size[1] - 4)),
                (bx1 + t_size[0] + 4, max(0, by1)),
                ppe_color,
                -1
            )
            cv2.putText(
                annotated,
                label,
                (bx1 + 2, max(10, by1 - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (20, 20, 20),
                1,
                cv2.LINE_AA
            )

    # 2. Vẽ khung công nhân và nhãn đánh giá suy luận
    for ev in evaluations:
        color = STATUS_COLORS.get(ev.status, (120, 120, 120))
        px1, py1, px2, py2 = [int(v) for v in ev.person_bbox]

        # Giới hạn trong kích thước ảnh
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(img_w - 1, px2), min(img_h - 1, py2)

        # Vẽ khung công nhân dày
        cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 3, cv2.LINE_AA)

        # Vẽ vùng đầu ước lượng bằng viền đứt nét mờ để hiển thị logic kiểm tra
        hx1, hy1, hx2, hy2 = [int(v) for v in ev.head_bbox]
        hx1, hy1 = max(0, hx1), max(0, hy1)
        hx2, hy2 = min(img_w - 1, hx2), min(img_h - 1, hy2)
        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), color, 1, cv2.LINE_4)

        # Nhãn thông tin suy luận: Worker_ID [Rule_ID] - Risk
        tag_text = f"{ev.worker_id} [{ev.rule_id}] {ev.risk_level}"
        font_scale = 0.55
        thickness = 1
        (txt_w, txt_h), baseline = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        label_top = max(0, py1 - txt_h - 8)
        cv2.rectangle(
            annotated,
            (px1, label_top),
            (px1 + txt_w + 10, label_top + txt_h + 8),
            color,
            -1
        )
        cv2.putText(
            annotated,
            tag_text,
            (px1 + 5, label_top + txt_h + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return annotated


def filter_duplicate_boxes(boxes: List[DetectedObject], iou_thresh: float = 0.45) -> List[DetectedObject]:
    """Khử các bounding box bị trùng lặp (Non-Maximum Suppression) cho cùng một người."""
    persons = [b for b in boxes if b.cls_name == "person"]
    others = [b for b in boxes if b.cls_name != "person"]

    if not persons:
        return boxes

    # Sắp xếp person theo conf giảm dần
    persons.sort(key=lambda x: x.conf, reverse=True)
    kept_persons: List[DetectedObject] = []

    for p in persons:
        is_duplicate = False
        for kp in kept_persons:
            # Tính IoU giữa p và kp
            ix1 = max(p.bbox[0], kp.bbox[0])
            iy1 = max(p.bbox[1], kp.bbox[1])
            ix2 = min(p.bbox[2], kp.bbox[2])
            iy2 = min(p.bbox[3], kp.bbox[3])

            inter_w = max(0.0, ix2 - ix1)
            inter_h = max(0.0, iy2 - iy1)
            inter_area = inter_w * inter_h

            area1 = max(0.0, (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1]))
            area2 = max(0.0, (kp.bbox[2] - kp.bbox[0]) * (kp.bbox[3] - kp.bbox[1]))
            union_area = area1 + area2 - inter_area

            iou = (inter_area / union_area) if union_area > 0 else 0.0
            if iou > iou_thresh:
                is_duplicate = True
                break

        if not is_duplicate:
            kept_persons.append(p)

    return kept_persons + others


@app.post("/api/analyze", response_model=AnalysisResponse, summary="Phân tích ảnh và suy luận luật an toàn")
async def analyze_image(
    file: UploadFile = File(..., description="Ảnh cần phân tích (.jpg, .jpeg, .png, .webp)"),
    conf_ppe: float = Form(0.25, ge=0.01, le=1.0, description="Ngưỡng tin cậy cho trang bị PPE (Tùy chỉnh qua API)"),
    conf_person: float = Form(0.35, ge=0.01, le=1.0, description="Ngưỡng tin cậy cho người (Tùy chỉnh qua API, mặc định 0.35)"),
    iou_threshold: float = Form(0.45, ge=0.01, le=1.0, description="Ngưỡng IoU Non-Maximum Suppression chống trùng lặp khung hình")
):
    """Endpoint tiếp nhận ảnh, thực hiện YOLO inference và kích hoạt Rule Engine.
    
    Các tham số conf_ppe, conf_person và iou_threshold được mở cho các kỹ sư kiểm thử qua API/Swagger Docs.
    """
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Mô hình YOLO chưa sẵn sàng.")

    # Kiểm tra định dạng file
    allowed_content_types = ["image/jpeg", "image/png", "image/webp", "image/bmp"]
    if file.content_type not in allowed_content_types and not any(file.filename.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]):
        raise HTTPException(
            status_code=400,
            detail="Tập tin tải lên không phải là ảnh hợp lệ (.jpg, .jpeg, .png, .webp, .bmp)"
        )

    # Đọc dữ liệu ảnh vào bộ nhớ
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Không thể giải mã dữ liệu ảnh (cv2.imdecode trả về None).")
    except Exception as e:
        logger.error(f"Lỗi khi đọc file ảnh: {e}")
        raise HTTPException(status_code=400, detail=f"Không thể xử lý file ảnh tải lên: {str(e)}")

    img_h, img_w = image.shape[:2]

    # Chạy inference mô hình YOLO với NMS iou_threshold
    min_conf = min(conf_ppe, conf_person)
    try:
        results = model.predict(source=image, conf=min_conf, iou=iou_threshold, verbose=False)
    except Exception as e:
        logger.error(f"Lỗi inference YOLO: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi trong quá trình chạy mô hình AI: {str(e)}")

    # Trích xuất và phân loại đối tượng phát hiện
    raw_detections: List[DetectedObject] = []
    names = getattr(model, "names", CLASS_NAMES)

    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            for idx, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()

                # Tên class ánh xạ theo model hoặc dictionary mặc định
                cls_name = names.get(cls_id, CLASS_NAMES.get(cls_id, f"class_{cls_id}")).lower()

                # Lọc theo ngưỡng riêng biệt
                if cls_name == "person":
                    if conf < conf_person:
                        continue
                else:
                    if conf < conf_ppe:
                        continue

                raw_detections.append(DetectedObject(
                    obj_id=f"obj_{idx}_{cls_name}",
                    cls_id=cls_id,
                    cls_name=cls_name,
                    conf=round(conf, 3),
                    bbox=[round(float(c), 2) for c in xyxy]
                ))

    # Áp dụng NMS phụ trợ độc quyền cho Person để triệt tiêu hoàn toàn khung người đè nhau
    detections = filter_duplicate_boxes(raw_detections, iou_thresh=iou_threshold)

    # Kích hoạt Rule Engine
    evaluations, summary = rule_engine.evaluate(detections, img_w=img_w, img_h=img_h)

    # Vẽ bounding box và nhãn suy luận lên ảnh
    annotated_img = draw_visualizations(image, detections, evaluations)

    # Mã hóa ảnh kết quả sang định dạng Base64
    success, buffer = cv2.imencode(".jpg", annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not success:
        raise HTTPException(status_code=500, detail="Không thể mã hóa ảnh kết quả thành JPEG.")

    b64_string = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

    return AnalysisResponse(
        summary=summary,
        evaluations=evaluations,
        image_base64=b64_string
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
