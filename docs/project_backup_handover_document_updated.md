# Safety Helmet Detection with Rule-Based Violation Reasoning
## Project Backup & Handover Document (Updated)

> **Mục đích:** Lưu toàn bộ bối cảnh, quyết định kỹ thuật, kết quả tiền xử lý, kết quả huấn luyện thực nghiệm (YOLOv8n & YOLOv8s) và kế hoạch tiếp theo để chuyển sang cuộc trò chuyện mới mà không phải giải thích lại từ đầu.
>
> **Cập nhật lần cuối:** 15/11/2026  
> **Ngôn ngữ làm việc:** Tiếng Việt  
> **Trạng thái tổng quát:** Đã hoàn thành phần chính Bài 1; Đã hoàn thành Audit, cách ly dữ liệu nhiễu (465 empty labels) và chuyển đổi Polygon sang Bbox cho PPE Detection V1; Đã hoàn thành huấn luyện baseline với cả **YOLOv8n** (thành viên nhóm) và **YOLOv8s** (bản làm sạch dữ liệu); Sẵn sàng bước sang xây dựng **Rule Engine (Fact Base, Rule Base, Forward Chaining)** và **Demo Gradio**.

---

# 1. Tổng quan dự án

- **Tên đề tài:** *Safety Helmet Detection with Rule-Based Violation Reasoning*
- **Mục tiêu:** Xây dựng hệ thống phát hiện người và thiết bị bảo hộ lao động bằng Computer Vision, sau đó sử dụng hệ luật để suy luận trạng thái tuân thủ/vi phạm và giải thích kết quả.
- **Định hướng kỹ thuật:** YOLOv8 Object Detection + Object Association + Fact Base + Rule Base + Forward Chaining.
- **MVP:** Phân tích ảnh tĩnh do người dùng tải lên.
- **Hướng mở rộng:** Camera/live stream, nếu đủ thời gian và dữ liệu.
- **Môi trường làm việc dữ liệu & huấn luyện:** Google Colab (Tesla T4 GPU).
- **Ngôn ngữ triển khai:** Python.
- **Giao diện demo dự kiến:** Gradio (chưa triển khai).

## 1.1. Luồng xử lý hệ thống

1. Nhận ảnh đầu vào.
2. YOLOv8 phát hiện các đối tượng đã được huấn luyện (6 classes PPE).
3. Liên kết đối tượng (Object Association) và ước lượng vùng đầu (`HeadRegionEstimated`).
4. Chuẩn hóa kết quả detection & association thành các sự kiện trong **Fact Base**.
5. Áp dụng **Rule Base** bằng thuật toán **Forward Chaining**.
6. Trả về trạng thái đánh giá, `Rule ID` và lý do giải thích bằng ngôn ngữ tự nhiên.
7. Hiển thị ảnh kết quả kèm bounding box và bảng giải thích.

Ba trạng thái đầu ra dự kiến:
- `COMPLIANT`: Có đủ bằng chứng theo các quy tắc đã định nghĩa để xác định tuân thủ.
- `SUSPECTED_VIOLATION`: Có bằng chứng đủ tin cậy để nghi ngờ vi phạm theo quy tắc.
- `INSUFFICIENT_EVIDENCE`: Không đủ bằng chứng, vùng quan sát bị che khuất hoặc detection không đáng tin cậy.

**Nguyên tắc an toàn suy luận:** Không kết luận một người vi phạm chỉ vì mô hình không phát hiện Helmet. Cần kiểm tra vùng đầu có quan sát đủ rõ không, có phát hiện `person`/`vest` không, và `helmet` có được liên kết hình học với đúng người hay không.

---

# 2. Tiến độ học phần và báo cáo

## 2.1. Bài 1 – Tuần 4–5 (Đã hoàn thành phần chính)
- Cấu trúc báo cáo / mục lục.
- Phân công nhiệm vụ.
- Chương 1 – Cơ sở lý thuyết.

## 2.2. Bài 2 – Tuần 7–8 (Đang hoàn thiện các đầu ra cuối)
- [x] Dataset được chọn và chuẩn hóa (`PPE Detection V1` -> `ppe_v1_clean_bbox`).
- [x] YOLOv8 baseline đã huấn luyện (`YOLOv8n` và `YOLOv8s`).
- [x] Metrics trên tập Validation & Test (đã có đầy đủ của YOLOv8n và validation của YOLOv8s; cần chạy nốt cell đánh giá `split="test"` của YOLOv8s để hoàn tất bảng đối chiếu).
- [ ] Rule Engine bản đầu (Fact Base, Rule Base, Forward Chaining).
- [ ] Demo tích hợp detector và Rule Engine (Gradio).
- [ ] Bản nháp Chương 2 (Phương pháp) & Chương 3 (Thực nghiệm).

## 2.3. Bài 3 / Báo cáo cuối – Tuần 10–11
- Hoàn thiện chương trình end-to-end, kiểm thử edge cases, trình bày giới hạn hệ thống và hoàn thiện báo cáo.

---

# 3. Lịch sử và trạng thái các Dataset

## 3.1. SHWD (Đã tiền xử lý - Không ưu tiên)
- Đã chuyển 7.581 ảnh từ Pascal VOC XML sang YOLO format (`/content/dataset/SHWD_YOLO`) với 2 class (`hat`, `person`).
- Dừng sử dụng làm dataset chính do quá ít class PPE.

## 3.2. SH17 (Đã kiểm tra cấu trúc - Giữ làm tham chiếu)
- 8.099 ảnh, 17 class, 75.994 bounding boxes.
- Có class `head` (hữu ích để tham khảo khi thiết kế thuật toán ước lượng vùng đầu `HeadRegionEstimated`), nhưng số lượng `helmet` ít (~927 objects) và thiếu tập `test` tách biệt nên giữ làm dataset tham chiếu.

## 3.3. PPE Detection V1 (Dataset chính - Đã Audit, Làm sạch & Chuyển đổi Bbox)
- **Nguồn:** Kaggle (`beyzakucuk/ppe-detection-v1`) / Roboflow (`rbyz/ppe-6-classes-ntld9`, v6).
- **Thông số gốc:** 17.264 ảnh (13.949 train, 1.669 valid, 1.646 test), tất cả đều định dạng JPEG, RGB, kích thước đồng nhất `640 × 640`. Tổng 64.677 objects (64.537 Polygon + 140 Bbox), 0 lỗi định dạng tọa độ.
- **Mapping 6 classes (giữ nguyên xuyên suốt):**
  ```yaml
  names:
    0: boots
    1: gloves
    2: goggles
    3: helmet
    4: person
    5: vest
  ```

### Phát hiện quan trọng từ Pre-Conversion & Visual Audit (Đã thực hiện 04/11 – 15/11/2026)
1. **Phát hiện 465 ảnh Empty Label thực chất là `MISSING_ANNOTATION`:**
   - Trong dataset gốc có 465 file `.txt` rỗng (387 train, 32 valid, 46 test).
   - Qua tạo 10 trang Contact Sheet (50 ảnh/trang) và phân tích tiền tố tên file, toàn bộ 465 file này xuất phát từ **194 ảnh gốc** được Roboflow nhân bản (tiền tố `0_`, `1_`, `2_`, `3_`, `4_`, `5_`, `pos`, `Nein...`).
   - Kiểm tra trực quan cho thấy đây là các ảnh chụp cận cảnh mũ bảo hộ (`helmet`) hoặc thiết bị PPE nhưng **tác giả dataset quên không gán nhãn**. Nếu để lại trong tập huấn luyện, mô hình sẽ học sai rằng mũ bảo hộ là Background.
   - **Giải pháp đã thực hiện:** Cách ly toàn bộ 465 ảnh này sang thư mục riêng (`/content/ppe_v1_work/quarantine_empty_labels`), giữ nguyên 16.799 ảnh sạch có nhãn hợp lệ.
2. **Hiện tượng "Gán nhãn bộ phận" (Partial Annotation) trong tập dữ liệu:**
   - Phân tích phân bố cho thấy nhiều ảnh có gán nhãn `helmet`, `vest`, `goggles` nhưng **không gán nhãn `person`** cho người đang mặc chúng (ví dụ trong 1.637 ảnh Valid sạch, chỉ có 306 ảnh – chiếm 18.7% – có nhãn `person`, trong khi có tới 572 ảnh có `vest` và 534 ảnh có `helmet`).
   - **Ý nghĩa học thuật:** Đây là nguyên nhân chính khiến `Precision` và `mAP@50` của lớp `person` bị ghìm ở mức ~35%–38% (do mô hình phát hiện đúng người nhưng bị chấm là False Positive trên những ảnh thiếu nhãn `person`), và cũng là **luận điểm thực nghiệm quan trọng** chứng minh tại sao cần kết hợp **Rule-Based Reasoning** thay vì chỉ dựa vào detector đơn thuần.

### Dữ liệu sạch đã đóng gói & Backup trên Google Drive
- **Vị trí lưu trữ trên Google Drive:**
  `/content/drive/MyDrive/Safety_Helmet_Detection_Project/PPE_V1_Backup/`
  - `ppe_v1_clean_bbox.zip`: Bộ dữ liệu đã chuyển đổi toàn bộ Polygon sang YOLO Bbox và loại bỏ 465 ảnh rỗng (16.799 ảnh sạch: 13.562 train, 1.637 valid, 1.600 test).
  - `quarantine_empty_labels.zip`: 465 ảnh empty label bị cách ly làm bằng chứng thực nghiệm.
  - `reports/`: Các file báo cáo JSON/CSV và hình ảnh đối chiếu `PPE_V1_Polygon_vs_Bbox_Demo.png`.

---

# 4. Kết quả Huấn luyện & Đánh giá Thực nghiệm (YOLOv8)

Nhóm đã thực hiện huấn luyện và đối chứng 2 cấu hình mô hình YOLOv8 trên bộ dữ liệu PPE Detection V1:

## 4.1. Thí nghiệm 1: `YOLOv8n` Baseline (`SSV_AI` - Thành viên nhóm thực hiện)
- **Dữ liệu:** `SSV_Dataset_V1` (49,671 instances trong tập Train).
- **Cấu hình:** `yolov8n.pt` (3.2M params, 6.21 MB), `imgsz=640`, `batch=8`, `epochs=60` (Early stopping tại Epoch 50, **Best tại Epoch 30**), `optimizer=MuSGD`, `lr0=0.01`, `seed=42`, `auto_augment=randaugment`, `erasing=0.4`.
- **Trọng số lưu tại:** `Models/YOLOv8n_Baseline/best.pt`.
- **Kết quả tổng thể:**
  - **Best Validation (Epoch 30):** Precision = `53.19%` | Recall = `60.68%` | **mAP@50 = `52.57%`** | **mAP@50-95 = `38.22%`**.
  - **Độc lập trên Test:** Precision = `53.00%` | Recall = `59.84%` | **mAP@50 = `51.46%`** | **mAP@50-95 = `36.81%`**.
  - **F1-Score tối ưu:** `0.55` tại ngưỡng `conf = 0.235`.

## 4.2. Thí nghiệm 2: `YOLOv8s` Clean Dataset (`yolov8s_clean_v2` - Đã chạy trên Colab)
- **Dữ liệu:** `ppe_v1_clean_bbox` (13.562 ảnh train sạch, 1.637 ảnh valid sạch; đã cách ly 465 ảnh empty label).
- **Cấu hình:** `yolov8s.pt` (11,127,906 params, 22.5 MB), `imgsz=640`, `batch=32`, `epochs=60` (Early stopping tại Epoch 43, **Best tại Epoch 23**), `optimizer=auto`, `cos_lr=True`, `close_mosaic=10`, `auto_augment=None`, `erasing=0.0`, `seed=42`.
- **Trọng số lưu trên Google Drive:**
  `/content/drive/MyDrive/Safety_Helmet_Detection_Project/runs_yolov8/yolov8s_clean_v2/weights/best.pt`
- **Tốc độ suy luận (Tesla T4 GPU):** `0.3ms` preprocess, `5.6ms` inference, `2.4ms` postprocess (~8.3ms/ảnh).
- **Kết quả Validation tốt nhất (Epoch 23):**
  - **Overall (`all`):** Precision = `52.50%` | Recall = `58.50%` | **mAP@50 = `51.00%`** | **mAP@50-95 = `37.10%`**.

## 4.3. Bảng đối chiếu chi tiết theo từng Class (Validation)

| Class | YOLOv8n Val mAP@50 (%) | YOLOv8s Val mAP@50 (%) | YOLOv8s Val Precision (%) | YOLOv8s Val Recall (%) | YOLOv8s Val mAP@50-95 (%) | Ghi chú kỹ thuật |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **`goggles`** | 64.60% | **66.00%** | 61.30% | 66.40% | 46.20% | YOLOv8s đạt kết quả cao nhất ở lớp kính bảo hộ. |
| **`vest`** | **66.20%** | 63.00% | 55.70% | 65.40% | **48.90%** | Độ chính xác định vị khung (mAP@50-95) rất cao. |
| **`helmet`** | **58.90%** | 58.30% | 57.00% | **68.40%** | 42.40% | Recall cao (68.4%), ít bỏ sót mũ bảo hộ. |
| **`boots`** | **55.50%** | 52.90% | 56.20% | 52.50% | 38.40% | Ổn định trên cả 2 kiến trúc. |
| **`person`** | **38.10%** | 34.80% | 37.60% | 62.20% | 25.00% | Bị ảnh hưởng bởi nhãn bộ phận (chỉ 18.7% ảnh có nhãn `person`). |
| **`gloves`** | **32.10%** | 31.20% | 47.50% | 35.80% | 21.40% | Đối tượng nhỏ, diện tích hẹp và ít mẫu nhất. |
| **Trung bình** | **52.57%** | **51.00%** | **52.50%** | **58.50%** | **37.10%** | Hai mô hình hội tụ cùng ngưỡng trần dữ liệu (~51–52.5%). |

*Lưu ý cho báo cáo:* Việc `YOLOv8s` (11.1M tham số, tắt erasing/randaugment) đạt `train/box_loss = 0.6889` rất thấp nhưng `val mAP@50` dừng sớm ở Epoch 23 cho thấy mô hình lớn dễ học thuộc lòng (overfit) các nhiễu nhãn cục bộ của bộ dữ liệu hơn mô hình nhẹ `YOLOv8n` có bật regularization (`RandAugment` + `Erasing`). Nhóm có thể chọn **`YOLOv8s`** (cho môi trường GPU tốc độ 5.6ms) hoặc **`YOLOv8n`** (cho môi trường CPU nhẹ 6.21MB) để tích hợp vào pipeline suy luận.

---

# 5. Kế hoạch thực hiện tiếp theo (Giai đoạn C & D)

## Bước 1: Hoàn thiện số liệu Test của `YOLOv8s` (5 phút trên Colab)
- Chạy `best_model.val(data=..., split="test")` đối với `yolov8s_clean_v2/weights/best.pt` để lấy bảng chỉ số `Test` đầy đủ đưa vào Chương 3.

## Bước 2: Xây dựng Hệ thống Suy luận Luật (Rule Engine + Forward Chaining)
Thiết kế module Python độc lập nhận output detections từ YOLOv8 và xử lý qua 4 tầng:
1. **Detection Filtering:** Lọc theo ngưỡng confidence tối ưu (ví dụ `conf >= 0.25` cho PPE và `0.30` cho `person`).
2. **Spatial & Geometric Association:**
   - Từ mỗi bounding box `person(x1, y1, x2, y2)`, ước lượng vùng đầu `head_region` (ví dụ 25%–30% phần trên cùng của khung `person`) và vùng thân `torso_region` (phần giữa khung `person`).
   - Liên kết `helmet` với `person` dựa trên độ giao nhau (IoU / Containment) giữa `helmet` và `head_region`.
   - Liên kết `vest` với `person` dựa trên độ giao nhau với `torso_region`.
   - **Xử lý trường hợp thiếu nhãn `person` (Fallback Rule):** Nếu phát hiện `vest` hoặc cụm `helmet` + `goggles` ở tỷ lệ hợp lệ mà detector bỏ sót khung `person`, hệ thống có thể suy luận sự hiện diện của người lao động thay vì bỏ qua.
3. **Fact Base & Rule Base (Forward Chaining):**
   - Khởi tạo tập sự kiện (Facts): `PersonDetected`, `HeadRegionVisible`, `HelmetAssociated`, `VestAssociated`, `UnassociatedHelmet`...
   - Tập luật (Rules) kèm `Rule_ID`, mức độ ưu tiên và lời giải thích tiếng Việt:
     - `R01`: Phát hiện `person` + vùng đầu rõ ràng + có `helmet` liên kết + có `vest` liên kết $\rightarrow$ `COMPLIANT` (Tuân thủ đầy đủ bảo hộ cơ bản).
     - `R02`: Phát hiện `person` + vùng đầu rõ ràng + có `helmet` liên kết + thiếu `vest` $\rightarrow$ `SUSPECTED_VIOLATION` (Đội mũ bảo hộ nhưng thiếu áo phản quang).
     - `R03`: Phát hiện `person` + vùng đầu rõ ràng + không có `helmet` liên kết $\rightarrow$ `SUSPECTED_VIOLATION` (Vi phạm quy định đội mũ bảo hộ).
     - `R04`: Phát hiện `person` nhưng vùng đầu chạm sát mép trên ảnh / bị cắt cụt hoặc confidence quá thấp $\rightarrow$ `INSUFFICIENT_EVIDENCE` (Không đủ bằng chứng kết luận vùng đầu).
     - `R05`: Không phát hiện `person` nhưng phát hiện `vest` mà không có `helmet` phía trên $\rightarrow$ `SUSPECTED_VIOLATION` / `INSUFFICIENT_EVIDENCE` (Cảnh báo người mặc áo bảo hộ thiếu mũ).
4. **Giao diện Demo (Gradio):**
   - Cho phép tải ảnh lên, chọn trọng số (`YOLOv8n` hoặc `YOLOv8s`), chỉnh ngưỡng confidence, hiển thị ảnh vẽ bbox (phân màu Xanh = Tuân thủ, Đỏ = Vi phạm, Vàng = Không đủ bằng chứng) kèm bảng chi tiết `Fact Base`, `Rule ID` kích hoạt và giải thích lý do.

---

# 6. Quy tắc làm việc trong cuộc trò chuyện mới

1. Trả lời bằng tiếng Việt, chia code Google Colab thành từng cell rõ ràng, độc lập.
2. Không ghi đè dataset gốc; sử dụng trực tiếp file backup `ppe_v1_clean_bbox.zip` và trọng số `best.pt` đã lưu trên Google Drive (`/content/drive/MyDrive/Safety_Helmet_Detection_Project/`).
3. Không bịa đặt metrics; mọi số liệu trích dẫn trong báo cáo phải bám sát kết quả thực nghiệm ở Mục 4.
4. Trọng tâm tiếp theo là hoàn thiện **Object Association + Fact Base + Rule Base (Forward Chaining) + Gradio Demo** để hoàn thành Chương 2, Chương 3 và sản phẩm demo cuối kỳ.

---

# 7. Prompt mở đầu cho cuộc trò chuyện mới

Sao chép đoạn sau kèm file Markdown này vào cuộc trò chuyện mới:

> Tôi đang thực hiện đề tài **Safety Helmet Detection with Rule-Based Violation Reasoning**. Hãy đọc kỹ file backup Markdown đính kèm và dùng nó làm tài liệu bối cảnh chính của dự án.
>
> Trạng thái hiện tại:
> - Đã hoàn thành Audit, cách ly 465 ảnh Empty Label lỗi và chuyển đổi Polygon sang Bbox cho dataset `PPE Detection V1` (`ppe_v1_clean_bbox.zip` đã lưu trên Google Drive).
> - Đã hoàn thành huấn luyện 2 mô hình đối chứng là `YOLOv8n` (`mAP@50 = 51.46%` trên Test) và `YOLOv8s` (`mAP@50 = 51.00%` trên Val, trọng số `best.pt` đã lưu trên Google Drive).
> - Đã phân tích được hiện tượng thiếu nhãn `person` cục bộ trong dataset (chỉ 18.7% ảnh có nhãn `person`), từ đó khẳng định vai trò thiết yếu của hệ luật suy diễn.
>
> Hãy hướng dẫn tôi bước tiếp theo:
> 1. Chạy nhanh cell đánh giá tập `Test` cho `YOLOv8s` để chốt số liệu Chương 3.
> 2. Xây dựng hoàn chỉnh module **Object Association + Fact Base + Rule Base (Forward Chaining)** với các trạng thái `COMPLIANT`, `SUSPECTED_VIOLATION`, `INSUFFICIENT_EVIDENCE` kèm `Rule ID` và giải thích chi tiết.
> 3. Xây dựng giao diện demo **Gradio** trên Google Colab để chạy kiểm thử thực tế trên ảnh.