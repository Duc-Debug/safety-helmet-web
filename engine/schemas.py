from typing import List
from pydantic import BaseModel, Field


class DetectedObject(BaseModel):
    """Mô hình dữ liệu cho một đối tượng phát hiện từ YOLO."""
    obj_id: str = Field(..., description="Định danh duy nhất của đối tượng")
    cls_id: int = Field(..., description="ID lớp phát hiện")
    cls_name: str = Field(..., description="Tên lớp (helmet, person, vest, ...)")
    conf: float = Field(..., description="Độ tin cậy của phát hiện")
    bbox: List[float] = Field(..., description="Tọa độ bounding box [x1, y1, x2, y2]")


class WorkerEvaluation(BaseModel):
    """Kết quả suy luận an toàn cho từng công nhân."""
    worker_id: str = Field(..., description="Mã định danh công nhân")
    person_bbox: List[float] = Field(..., description="Tọa độ hộp người [x1, y1, x2, y2]")
    head_bbox: List[float] = Field(..., description="Tọa độ ước tính vùng đầu [x1, y1, x2, y2]")
    facts: List[str] = Field(default_factory=list, description="Danh sách các sự kiện đã kiểm chứng")
    status: str = Field(..., description="Mã trạng thái quy tắc (COMPLIANT, DANGER_NO_HELMET, ...)")
    risk_level: str = Field(..., description="Mức độ rủi ro (AN TOÀN, NGUY HIỂM CAO, ...)")
    rule_id: str = Field(..., description="Mã luật áp dụng (R01, R02, R03, R04, R05)")
    explanation: str = Field(..., description="Giải thích kết quả suy luận logic")


class SummaryMetrics(BaseModel):
    """Thống kê tổng hợp toàn bộ khung cảnh."""
    total_workers: int = Field(0, description="Tổng số người phát hiện/nội suy")
    valid_evaluated: int = Field(0, description="Số người đủ dữ kiện để đánh giá")
    helmet_ok: int = Field(0, description="Số người đội mũ bảo hộ hợp lệ")
    danger_no_helmet: int = Field(0, description="Số người không đội mũ bảo hộ nguy hiểm")
    insufficient_evidence: int = Field(0, description="Số người không đủ bằng chứng")
    helmet_compliance_rate: float = Field(0.0, description="Tỉ lệ tuân thủ mũ bảo hộ (%)")
    danger_rate: float = Field(0.0, description="Tỉ lệ nguy hiểm không đội mũ (%)")
    is_scene_dangerous: bool = Field(False, description="Cảnh báo hiện trường có nguy hiểm hay không")


class AnalysisResponse(BaseModel):
    """Phản hồi đầy đủ từ API phân tích."""
    summary: SummaryMetrics
    evaluations: List[WorkerEvaluation]
    image_base64: str
