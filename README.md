# Safety Helmet Detection with Rule-Based Violation Reasoning

Hệ thống thị giác máy tính kết hợp suy luận logic dựa trên luật (Computer Vision + Rule-Based Reasoning Engine) nhằm giám sát tuân thủ mũ bảo hộ và trang bị bảo hộ cá nhân (PPE) trên công trường xây dựng theo thời gian thực.

---

## 1. Điểm nổi bật & Kiến trúc hệ thống

- **Mô hình Hybrid AI:** Kết hợp mô hình Deep Learning **YOLOv8** (phát hiện đối tượng `helmet`, `person`, `vest`, `boots`, `gloves`, `goggles`) và **Rule Engine** suy diễn tiến (Forward Chaining) hoàn toàn minh bạch (Explainable AI).
- **Liên kết không gian hình học chống báo động giả (Spatial & Geometric Association):**
  - Thu hẹp chiều ngang vùng đầu (`HeadRegion`: co 15% mỗi bên ngang, lấy từ `y1 - 0.08h` đến `y1 + 0.30h`) tránh nhận nhầm mũ người bên cạnh khi đám đông đứng sát nhau.
  - Vùng thân áo (`TorsoRegion`: co 10% mỗi bên ngang, lấy từ `0.20h` đến `0.72h`).
  - Liên kết kép **Point-in-Box + Intersection**: Gán `helmet` gần tâm đầu nhất; gộp mọi mảnh `vest` (áo phanh ngực sinh 2 box trái/phải) vào người tương ứng để triệt tiêu box vest dư thừa.
- **Kiểm soát Fallback Rule nghiêm ngặt:** Trước khi suy luận `InferredPersonFromVest`, kiểm tra tâm vest có nằm trong hoặc cách mép dưới 15px của bất kỳ `person` nào không; nếu có thì bỏ qua hoàn toàn, chỉ tạo người ảo khi vest nằm ở không gian độc lập thực sự.
- **Cơ chế chống trùng lặp đa khung hình (NMS Anti-Duplication):** Tích hợp thuật toán NMS lọc kép (IoU = 0.45), tự động triệt tiêu các khung người trùng lặp khi người đứng gần nhau hoặc khi hạ độ nhạy mô hình.
- **Bảo vệ cấu hình trực quan:** Giao diện Web được khóa cố định ở bộ thông số chuẩn hóa tối ưu (`conf_ppe=0.25`, `conf_person=0.35`, `iou_threshold=0.45`), người dùng phổ thông không thể can thiệp sai. Các kỹ sư muốn tinh chỉnh sâu có thể sử dụng trực tiếp qua giao diện Swagger API (`/docs`).
- **Tập luật suy diễn ưu tiên (Ruleset):**
  - `R01`: Tuân thủ đầy đủ (Có mũ + Có áo phản quang) -> **AN TOÀN** (`COMPLIANT`)
  - `R02`: Có mũ bảo hộ nhưng thiếu áo phản quang -> **CẢNH BÁO NHẸ** (`WARNING_NO_VEST`)
  - `R03`: Thấy rõ người và đỉnh đầu nhưng KHÔNG ĐỘI MŨ -> **NGUY HIỂM CAO** (`DANGER_NO_HELMET`)
  - `R04`: Thấy áo bảo hộ nhưng vùng đầu phía trên KHÔNG ĐỘI MŨ -> **NGUY HIỂM CAO** (`DANGER_NO_HELMET`)
  - `R05`: Vùng đầu sát mép trên khung hình (`y1 <= 5px`) hoặc bị che khuất -> **KHÔNG ĐỦ BẰNG CHỨNG** (`INSUFFICIENT_EVIDENCE`)

---

## 2. Yêu cầu cài đặt

Khuyến nghị sử dụng **Python 3.9 - 3.11**.

Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

---

## 3. Khởi chạy ứng dụng

Chạy máy chủ backend FastAPI bằng lệnh Uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Sau khi máy chủ khởi động thành công:
- **Giao diện Tiêu chuẩn (Vận hành chuẩn hóa - Khóa chỉnh thanh trượt):** [http://localhost:8000](http://localhost:8000)
- **Giao diện Tinh chỉnh chuyên sâu (Tuning Lab với đầy đủ Sliders):** [http://localhost:8000/tuning](http://localhost:8000/tuning)
- **Tài liệu tương tác API Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 4. Cấu trúc mã nguồn

```
safety-helmet-web/
├── engine/
│   ├── __init__.py           # Export các module và schemas
│   ├── schemas.py            # Pydantic schemas (DetectedObject, WorkerEvaluation, SummaryMetrics)
│   └── rule_engine.py        # Lõi suy diễn luật (Spatial Matching, Fact Base, Forward Chaining)
├── static/
│   └── index.html            # Giao diện Web Tiếng Việt (TailwindCSS, KPI, Banner cảnh báo, Rule Trace)
├── weights/
│   └── best.pt               # Trọng số YOLOv8 đã fine-tune (tự động fallback yolov8n.pt nếu thiếu)
├── main.py                   # FastAPI app, API /api/analyze, OpenCV Visualizer & Base64 encoder
├── requirements.txt          # Danh sách thư viện phụ thuộc
└── README.md                 # Tài liệu hướng dẫn sử dụng
```
