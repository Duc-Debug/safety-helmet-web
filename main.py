import os
import io
import time
import base64
import logging
from typing import List, Dict, Any, Tuple
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
    ModelBenchmarkResult,
    ComparisonResponse,
)
from engine.rule_engine import SafetyRuleEngine, CLASS_NAMES
from engine.yolov5_wrapper import YOLOv5Wrapper

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("safety-helmet-service")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Cấu hình 3 kiến trúc mô hình (YOLOv8n, YOLOv8s, YOLOv5) theo 3 thư mục riêng biệt
AVAILABLE_MODELS = {
    "yolov8s": {
        "title": "YOLOv8s (Fine-Tuned Small)",
        "paths": [
            os.path.join("weights", "yolov8s", "best.pt"),
            os.path.join("weights", "yolov8s", "yolov8s.pt"),
            os.path.join("weights", "best.pt"),
        ],
        "fallback": "yolov8s.pt",
        "description": "High accuracy on small PPE items, balanced speed"
    },
    "yolov8n": {
        "title": "YOLOv8n (Ultra-Fast Nano)",
        "paths": [
            os.path.join("weights", "yolov8n", "best.pt"),
            os.path.join("weights", "yolov8n", "yolov8n.pt"),
            os.path.join("weights", "yolov8n.pt"),
        ],
        "fallback": "yolov8n.pt",
        "description": "Lightweight model, fastest edge camera inference"
    },
    "yolov5": {
        "title": "YOLOv5 (Classic Baseline)",
        "paths": [
            os.path.join("weights", "yolov5", "best.pt"),
            os.path.join("weights", "yolov5", "yolov5s.pt"),
            os.path.join("weights", "yolov5", "yolov5.pt"),
            os.path.join("weights", "yolov5s.pt"),
        ],
        "fallback": "yolov5su.pt",
        "description": "Standard industrial benchmark architecture"
    }
}

# Cache mô hình đã nạp vào bộ nhớ RAM
LOADED_MODELS: Dict[str, YOLO] = {}
rule_engine = SafetyRuleEngine()


def get_yolo_model(model_key: str = "yolov8s") -> Tuple[YOLO, str]:
    """Lấy mô hình YOLO từ cache RAM hoặc nạp mới từ các thư mục weights tương ứng."""
    key = model_key.lower().strip()
    if key not in AVAILABLE_MODELS:
        key = "yolov8s"

    if key in LOADED_MODELS:
        return LOADED_MODELS[key], key

    cfg = AVAILABLE_MODELS[key]

    # Quét qua danh sách các file ứng viên trong thư mục weights/<model_key>/
    for candidate_path in cfg.get("paths", []):
        if os.path.exists(candidate_path):
            logger.info(f"Loading model '{key}' from disk: {candidate_path}")
            try:
                if key == "yolov5":
                    try:
                        m = YOLO(candidate_path)
                    except Exception as yolo_err:
                        logger.info(f"YOLOv8 engine không tải được checkpoint YOLOv5 ({yolo_err}). Chuyển sang YOLOv5Wrapper...")
                        m = YOLOv5Wrapper(candidate_path)
                else:
                    m = YOLO(candidate_path)
                LOADED_MODELS[key] = m
                logger.info(f"Loaded '{key}' successfully from {candidate_path}")
                return m, key
            except Exception as e:
                logger.error(f"Error loading {candidate_path}: {e}")

    # Fallback nếu chưa có file đặt trong thư mục
    logger.info(f"No custom local file found for '{key}'. Loading via fallback: {cfg['fallback']}")
    m = YOLO(cfg["fallback"])
    LOADED_MODELS[key] = m
    return m, key


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời ứng dụng FastAPI: tiền nạp mô hình chính (YOLOv8s) khi khởi động."""
    logger.info("Khởi động hệ thống Safety Helmet Vision...")
    try:
        get_yolo_model("yolov8s")
    except Exception as e:
        logger.warning(f"Chưa thể tiền nạp yolov8s khi khởi động: {e}. Sẽ nạp on-demand khi có request.")
    yield
    logger.info("Ứng dụng đang dừng hoạt động...")


app = FastAPI(
    title="Safety Helmet Detection & Rule-Based Reasoning API",
    description="Hệ thống nhận diện mũ bảo hộ và trang bị PPE kết hợp lập luận suy diễn luật",
    version="1.1.0",
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

# Mount thư mục static
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", summary="Standard User View")
async def serve_index():
    """Phục vụ file giao diện web chính static/index.html (Chế độ tiêu chuẩn đã thu gọn)."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_file):
        raise HTTPException(status_code=404, detail="File static/index.html not found.")
    return FileResponse(index_file)


@app.get("/tuning", summary="Advanced Tuning Studio")
async def serve_tuning():
    """Phục vụ file giao diện tinh chỉnh tham số chuyên sâu static/tuning.html."""
    tuning_file = os.path.join(STATIC_DIR, "tuning.html")
    if not os.path.exists(tuning_file):
        raise HTTPException(status_code=404, detail="File static/tuning.html not found.")
    return FileResponse(tuning_file)


@app.get("/compare", summary="Model Benchmark & Comparison Page")
async def serve_compare():
    """Phục vụ file giao diện so sánh trực quan đa mô hình static/compare.html."""
    compare_file = os.path.join(STATIC_DIR, "compare.html")
    if not os.path.exists(compare_file):
        raise HTTPException(status_code=404, detail="File static/compare.html not found.")
    return FileResponse(compare_file)


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


def draw_visualizations(
    image: np.ndarray,
    detections: List[DetectedObject],
    evaluations: List[WorkerEvaluation]
) -> np.ndarray:
    """Vẽ bounding boxes của PPE và công nhân kèm nhãn suy luận lên ảnh."""
    annotated = image.copy()
    img_h, img_w = annotated.shape[:2]

    STATUS_COLORS: Dict[str, tuple] = {
        "COMPLIANT": (0, 200, 0),             # Xanh lá
        "WARNING_NO_VEST": (0, 165, 255),      # Cam
        "DANGER_NO_HELMET": (0, 0, 255),       # Đỏ
        "INSUFFICIENT_EVIDENCE": (160, 160, 160)  # Xám
    }

    # 1. Vẽ các vật thể PPE nhỏ (helmet, vest, boots, gloves, goggles)
    ppe_color = (0, 220, 255)
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

        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(img_w - 1, px2), min(img_h - 1, py2)

        cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 3, cv2.LINE_AA)

        hx1, hy1, hx2, hy2 = [int(v) for v in ev.head_bbox]
        hx1, hy1 = max(0, hx1), max(0, hy1)
        hx2, hy2 = min(img_w - 1, hx2), min(img_h - 1, hy2)
        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), color, 1, cv2.LINE_4)

        tag_text = f"{ev.worker_id} [{ev.rule_id}] {ev.risk_level}"
        font_scale = 0.52
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


def run_pipeline(
    image: np.ndarray,
    yolo_model: YOLO,
    conf_ppe: float,
    conf_person: float,
    iou_thresh: float
) -> Tuple[SummaryMetrics, List[WorkerEvaluation], str, float]:
    """Chạy toàn bộ pipeline từ YOLO inference -> Rule Engine -> Visualization."""
    img_h, img_w = image.shape[:2]
    min_conf = min(conf_ppe, conf_person)

    # Đo thời gian suy luận AI
    start_t = time.perf_counter()
    results = yolo_model.predict(source=image, conf=min_conf, iou=iou_thresh, verbose=False)
    inference_time_ms = round((time.perf_counter() - start_t) * 1000, 1)

    raw_detections: List[DetectedObject] = []
    names = getattr(yolo_model, "names", CLASS_NAMES)

    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            for idx, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()

                cls_name = names.get(cls_id, CLASS_NAMES.get(cls_id, f"class_{cls_id}")).lower()

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

    # NMS phụ trợ khử trùng lặp người
    detections = filter_duplicate_boxes(raw_detections, iou_thresh=iou_thresh)

    # Kích hoạt Rule Engine
    evaluations, summary = rule_engine.evaluate(detections, img_w=img_w, img_h=img_h)

    # Vẽ bounding boxes
    annotated_img = draw_visualizations(image, detections, evaluations)

    # Encode Base64
    success, buffer = cv2.imencode(".jpg", annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not success:
        raise HTTPException(status_code=500, detail="Cannot encode resulting image to JPEG.")

    b64_string = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    return summary, evaluations, b64_string, inference_time_ms


@app.post("/api/analyze", response_model=AnalysisResponse, summary="Analyze image and perform safety rule reasoning")
async def analyze_image(
    file: UploadFile = File(..., description="Image to analyze (.jpg, .jpeg, .png, .webp)"),
    model_type: str = Form("yolov8s", description="Selected model architecture: yolov8s, yolov8n, yolov5"),
    conf_ppe: float = Form(0.25, ge=0.01, le=1.0, description="Confidence threshold for PPE gear"),
    conf_person: float = Form(0.35, ge=0.01, le=1.0, description="Confidence threshold for person"),
    iou_threshold: float = Form(0.45, ge=0.01, le=1.0, description="IoU threshold for Non-Maximum Suppression (NMS)")
):
    """Phân tích ảnh với kiến trúc mô hình được chọn (YOLOv8s, YOLOv8n, hoặc YOLOv5)."""
    # Kiểm tra định dạng file
    allowed_content_types = ["image/jpeg", "image/png", "image/webp", "image/bmp"]
    if file.content_type not in allowed_content_types and not any(file.filename.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image format (.jpg, .jpeg, .png, .webp, .bmp)"
        )

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image data.")
    except Exception as e:
        logger.error(f"Error reading image file: {e}")
        raise HTTPException(status_code=400, detail=f"Cannot process uploaded image: {str(e)}")

    selected_model, active_key = get_yolo_model(model_type)
    summary, evaluations, b64_string, latency_ms = run_pipeline(
        image, selected_model, conf_ppe, conf_person, iou_threshold
    )

    return AnalysisResponse(
        summary=summary,
        evaluations=evaluations,
        image_base64=b64_string,
        model_used=active_key,
        inference_time_ms=latency_ms
    )


@app.post("/api/compare", response_model=ComparisonResponse, summary="Run benchmark comparison on all 3 models")
async def compare_models(
    file: UploadFile = File(..., description="Image to benchmark (.jpg, .jpeg, .png, .webp)"),
    conf_ppe: float = Form(0.25, ge=0.01, le=1.0),
    conf_person: float = Form(0.35, ge=0.01, le=1.0),
    iou_threshold: float = Form(0.45, ge=0.01, le=1.0)
):
    """Chạy đồng thời cả 3 kiến trúc mô hình (YOLOv8n, YOLOv8s, YOLOv5) trên cùng 1 ảnh để so sánh."""
    allowed_content_types = ["image/jpeg", "image/png", "image/webp", "image/bmp"]
    if file.content_type not in allowed_content_types and not any(file.filename.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image format (.jpg, .jpeg, .png, .webp, .bmp)"
        )

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image data.")
    except Exception as e:
        logger.error(f"Error reading image file: {e}")
        raise HTTPException(status_code=400, detail=f"Cannot process uploaded image: {str(e)}")

    benchmark_results: Dict[str, ModelBenchmarkResult] = {}
    target_models = ["yolov8n", "yolov8s", "yolov5"]

    for m_key in target_models:
        try:
            m_obj, resolved_key = get_yolo_model(m_key)
            summary, evaluations, b64_str, latency = run_pipeline(
                image, m_obj, conf_ppe, conf_person, iou_threshold
            )
            title = AVAILABLE_MODELS.get(m_key, {}).get("title", m_key.upper())
            benchmark_results[m_key] = ModelBenchmarkResult(
                model_key=m_key,
                model_title=title,
                inference_time_ms=latency,
                summary=summary,
                evaluations=evaluations,
                image_base64=b64_str
            )
        except Exception as e:
            logger.error(f"Error benchmarking model {m_key}: {e}")
            # Fallback nếu 1 model gặp trục trặc
            continue

    if not benchmark_results:
        raise HTTPException(status_code=500, detail="Failed to run comparison on models.")

    return ComparisonResponse(results=benchmark_results)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
