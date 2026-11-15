from typing import List, Dict, Tuple, Optional
import math
from .schemas import DetectedObject, WorkerEvaluation, SummaryMetrics

CLASS_NAMES: Dict[int, str] = {
    0: 'boots',
    1: 'gloves',
    2: 'goggles',
    3: 'helmet',
    4: 'person',
    5: 'vest'
}


class SafetyRuleEngine:
    """Lõi suy diễn luật an toàn lao động (Rule-Based Reasoning Engine).

    Thực hiện phân tích không gian hình học (Spatial & Geometric Association),
    khởi tạo cơ sở sự kiện (Fact Base) và áp dụng chuỗi suy diễn tiến (Forward Chaining)
    để phân loại rủi ro và đánh giá tuân thủ trang bị bảo hộ cá nhân (PPE).
    """

    HELMET_HEAD_THRESHOLD: float = 0.35
    VEST_TORSO_THRESHOLD: float = 0.35
    INFERRED_HELMET_HEAD_THRESHOLD: float = 0.30
    CLIPPED_BORDER_PIXELS: float = 5.0

    @staticmethod
    def calculate_intersection_coverage(item_bbox: List[float], target_region: List[float]) -> float:
        """Tính tỉ lệ phần diện tích của item nằm lọt bên trong target region.

        coverage = intersection_area / item_area
        """
        ix1 = max(item_bbox[0], target_region[0])
        iy1 = max(item_bbox[1], target_region[1])
        ix2 = min(item_bbox[2], target_region[2])
        iy2 = min(item_bbox[3], target_region[3])

        inter_w = max(0.0, ix2 - ix1)
        inter_h = max(0.0, iy2 - iy1)
        intersection_area = inter_w * inter_h

        item_w = max(0.0, item_bbox[2] - item_bbox[0])
        item_h = max(0.0, item_bbox[3] - item_bbox[1])
        item_area = item_w * item_h

        if item_area <= 1e-6:
            return 0.0

        return intersection_area / item_area

    @classmethod
    def estimate_person_regions(cls, bbox: List[float], img_h: Optional[float] = None) -> Tuple[List[float], List[float]]:
        """Ước tính vùng đầu (HeadRegion) và vùng thân áo (TorsoRegion) từ khung người.

        - HeadRegion: 28% chiều cao trên cùng của person, mở rộng nhẹ 5% lên phía trên để bao trọn mũ.
        - TorsoRegion: từ 22% đến 68% chiều cao của person.
        """
        x1, y1, x2, y2 = bbox
        h = max(0.0, y2 - y1)

        head_y1 = max(0.0, y1 - 0.05 * h)
        head_y2 = y1 + 0.28 * h
        head_bbox = [x1, head_y1, x2, head_y2]

        torso_y1 = y1 + 0.22 * h
        torso_y2 = y1 + 0.68 * h
        torso_bbox = [x1, torso_y1, x2, torso_y2]

        return head_bbox, torso_bbox

    @classmethod
    def infer_person_from_vest(cls, vest_bbox: List[float], img_w: Optional[float] = None, img_h: Optional[float] = None) -> Tuple[List[float], List[float]]:
        """Nội suy khung người và vùng đầu giả định từ chiếc áo bảo hộ khi detector bỏ sót person.

        Vest thông thường chiếm ~46% chiều cao cơ thể người.
        """
        vx1, vy1, vx2, vy2 = vest_bbox
        vw = max(0.0, vx2 - vx1)
        vh = max(0.0, vy2 - vy1)

        # Ước lượng chiều cao người từ chiều cao vest
        est_person_h = vh / 0.46

        # Vùng đầu giả định nằm ngay trên áo vest
        head_y1 = max(0.0, vy1 - 0.55 * vh)
        head_y2 = vy1 + 0.15 * vh
        head_x1 = max(0.0, vx1 - 0.05 * vw)
        head_x2 = vx2 + 0.05 * vw
        if img_w is not None:
            head_x2 = min(img_w, head_x2)
        head_bbox = [head_x1, head_y1, head_x2, head_y2]

        # Khung người nội suy giả định
        px1 = max(0.0, vx1 - 0.15 * vw)
        px2 = vx2 + 0.15 * vw
        if img_w is not None:
            px2 = min(img_w, px2)
        py1 = head_y1
        py2 = vy2 + 0.85 * vh
        if img_h is not None:
            py2 = min(img_h, py2)
        inferred_person_bbox = [px1, py1, px2, py2]

        return inferred_person_bbox, head_bbox

    def evaluate(self, detections: List[DetectedObject], img_w: Optional[float] = None, img_h: Optional[float] = None) -> Tuple[List[WorkerEvaluation], SummaryMetrics]:
        """Thực hiện liên kết không gian và suy diễn luật trên toàn bộ đối tượng phát hiện."""
        persons = [d for d in detections if d.cls_name == 'person']
        helmets = [d for d in detections if d.cls_name == 'helmet']
        vests = [d for d in detections if d.cls_name == 'vest']

        # Dữ liệu chuẩn bị cho từng người
        # worker_candidates: list of dict(type, bbox, head_bbox, torso_bbox, obj_id)
        worker_candidates: List[Dict] = []
        for idx, p in enumerate(persons):
            head_box, torso_box = self.estimate_person_regions(p.bbox, img_h)
            worker_candidates.append({
                "worker_id": f"Worker_{idx + 1:02d}",
                "is_inferred": False,
                "person_bbox": p.bbox,
                "head_bbox": head_box,
                "torso_bbox": torso_box,
                "associated_helmet": None,
                "associated_vest": None,
                "source_obj_id": p.obj_id
            })

        # 1. Liên kết Vest với TorsoRegion của Persons (Đảm bảo 1 vest chỉ gán tối đa 1 người)
        vest_matches: List[Tuple[float, int, int]] = []  # (coverage, candidate_idx, vest_idx)
        for c_idx, cand in enumerate(worker_candidates):
            for v_idx, vest in enumerate(vests):
                cov = self.calculate_intersection_coverage(vest.bbox, cand["torso_bbox"])
                if cov >= self.VEST_TORSO_THRESHOLD:
                    vest_matches.append((cov, c_idx, v_idx))

        # Sắp xếp giảm dần theo độ bao phủ để ưu tiên ghép tốt nhất
        vest_matches.sort(key=lambda x: x[0], reverse=True)
        assigned_candidates_vest = set()
        assigned_vests = set()

        for cov, c_idx, v_idx in vest_matches:
            if c_idx not in assigned_candidates_vest and v_idx not in assigned_vests:
                worker_candidates[c_idx]["associated_vest"] = vests[v_idx]
                assigned_candidates_vest.add(c_idx)
                assigned_vests.add(v_idx)

        # 2. Liên kết Helmet với HeadRegion của Persons (Đảm bảo 1 helmet chỉ gán tối đa 1 người)
        helmet_matches: List[Tuple[float, int, int]] = []
        for c_idx, cand in enumerate(worker_candidates):
            for h_idx, helmet in enumerate(helmets):
                h_cov = self.calculate_intersection_coverage(helmet.bbox, cand["head_bbox"])
                if h_cov >= self.HELMET_HEAD_THRESHOLD:
                    helmet_matches.append((h_cov, c_idx, h_idx))

        helmet_matches.sort(key=lambda x: x[0], reverse=True)
        assigned_candidates_helmet = set()
        assigned_helmets = set()

        for cov, c_idx, h_idx in helmet_matches:
            if c_idx not in assigned_candidates_helmet and h_idx not in assigned_helmets:
                worker_candidates[c_idx]["associated_helmet"] = helmets[h_idx]
                assigned_candidates_helmet.add(c_idx)
                assigned_helmets.add(h_idx)

        # 3. Fallback Rule: Xử lý các vest chưa được gán (detector bỏ sót khung person)
        unassigned_vests = [v for idx, v in enumerate(vests) if idx not in assigned_vests]
        for v in unassigned_vests:
            inferred_p_box, inferred_h_box = self.infer_person_from_vest(v.bbox, img_w, img_h)
            
            # Tìm helmet chưa được gán có coverage >= 0.30 với vùng đầu nội suy
            best_helmet = None
            best_cov = -1.0
            best_h_idx = -1

            for h_idx, helmet in enumerate(helmets):
                if h_idx in assigned_helmets:
                    continue
                cov = self.calculate_intersection_coverage(helmet.bbox, inferred_h_box)
                if cov >= self.INFERRED_HELMET_HEAD_THRESHOLD and cov > best_cov:
                    best_cov = cov
                    best_helmet = helmet
                    best_h_idx = h_idx

            if best_helmet is not None and best_h_idx >= 0:
                assigned_helmets.add(best_h_idx)

            worker_id = f"Worker_{len(worker_candidates) + 1:02d}"
            worker_candidates.append({
                "worker_id": worker_id,
                "is_inferred": True,
                "person_bbox": inferred_p_box,
                "head_bbox": inferred_h_box,
                "torso_bbox": v.bbox,
                "associated_helmet": best_helmet,
                "associated_vest": v,
                "source_obj_id": v.obj_id
            })

        # 4. Áp dụng Fact Base & Rule Base cho từng worker
        evaluations: List[WorkerEvaluation] = []
        for cand in worker_candidates:
            facts: List[str] = []

            # Fact: Nguồn gốc đối tượng
            if cand["is_inferred"]:
                facts.append("InferredPersonFromVest")
            else:
                facts.append("PersonDetected")

            # Fact: Vùng đầu rõ ràng hay bị cắt / khuất sát mép trên
            head_y1 = cand["head_bbox"][1]
            if head_y1 > self.CLIPPED_BORDER_PIXELS:
                facts.append("HeadRegionClear")
            else:
                facts.append("HeadRegionOccludedOrClipped")

            # Fact: Liên kết bảo hộ
            has_helmet = cand["associated_helmet"] is not None
            has_vest = cand["associated_vest"] is not None

            if has_helmet:
                facts.append("HelmetAssociated")
            if has_vest:
                facts.append("VestAssociated")

            # Áp dụng tập luật theo độ ưu tiên:
            status = ""
            risk_level = ""
            rule_id = ""
            explanation = ""

            is_clipped = "HeadRegionOccludedOrClipped" in facts
            is_clear = "HeadRegionClear" in facts

            # Nếu có nón bảo hộ, dù đầu sát mép trên vẫn chứng minh được an toàn
            if has_helmet and has_vest:
                # Rule R01
                rule_id = "R01"
                status = "COMPLIANT"
                risk_level = "AN TOÀN"
                explanation = "Tuân thủ đầy đủ: Đã đội mũ bảo hộ đúng vị trí vùng đầu và mặc áo phản quang."
            elif has_helmet and not has_vest:
                # Rule R02
                rule_id = "R02"
                status = "WARNING_NO_VEST"
                risk_level = "CẢNH BÁO NHẸ"
                explanation = "Đã đội mũ bảo hộ an toàn, nhưng thiếu áo phản quang bảo hộ."
            elif not has_helmet and is_clipped:
                # Rule R05
                rule_id = "R05"
                status = "INSUFFICIENT_EVIDENCE"
                risk_level = "KHÔNG ĐỦ BẰNG CHỨNG"
                explanation = "Vùng đầu nằm sát mép trên khung hình hoặc bị che khuất, chưa đủ cơ sở kết luận vi phạm."
            elif not has_helmet and is_clear:
                if not cand["is_inferred"]:
                    # Rule R03
                    rule_id = "R03"
                    status = "DANGER_NO_HELMET"
                    risk_level = "NGUY HIỂM CAO"
                    explanation = "VI PHẠM NGHIÊM TRỌNG: Phát hiện người lao động quan sát rõ vùng đầu nhưng KHÔNG ĐỘI MŨ BẢO HỘ."
                else:
                    # Rule R04
                    rule_id = "R04"
                    status = "DANGER_NO_HELMET"
                    risk_level = "NGUY HIỂM CAO"
                    explanation = "VI PHẠM NGHIÊM TRỌNG (Suy luận từ áo): Phát hiện người mặc áo bảo hộ nhưng vùng đầu phía trên KHÔNG CÓ MŨ."
            else:
                # Fallback an toàn (tránh trường hợp ngoại lai không khớp)
                rule_id = "R05"
                status = "INSUFFICIENT_EVIDENCE"
                risk_level = "KHÔNG ĐỦ BẰNG CHỨNG"
                explanation = "Dữ liệu vị trí không đủ điều kiện phân loại rõ ràng."

            evaluations.append(WorkerEvaluation(
                worker_id=cand["worker_id"],
                person_bbox=[round(coord, 2) for coord in cand["person_bbox"]],
                head_bbox=[round(coord, 2) for coord in cand["head_bbox"]],
                facts=facts,
                status=status,
                risk_level=risk_level,
                rule_id=rule_id,
                explanation=explanation
            ))

        # 5. Tính toán SummaryMetrics
        total_workers = len(evaluations)
        insufficient_ev = sum(1 for e in evaluations if e.status == "INSUFFICIENT_EVIDENCE")
        valid_evaluated = total_workers - insufficient_ev
        helmet_ok = sum(1 for e in evaluations if e.status in ["COMPLIANT", "WARNING_NO_VEST"])
        danger_no_helmet = sum(1 for e in evaluations if e.status == "DANGER_NO_HELMET")

        helmet_compliance_rate = round((helmet_ok / valid_evaluated * 100), 1) if valid_evaluated > 0 else 0.0
        danger_rate = round((danger_no_helmet / valid_evaluated * 100), 1) if valid_evaluated > 0 else 0.0
        is_scene_dangerous = danger_no_helmet > 0

        summary = SummaryMetrics(
            total_workers=total_workers,
            valid_evaluated=valid_evaluated,
            helmet_ok=helmet_ok,
            danger_no_helmet=danger_no_helmet,
            insufficient_evidence=insufficient_ev,
            helmet_compliance_rate=helmet_compliance_rate,
            danger_rate=danger_rate,
            is_scene_dangerous=is_scene_dangerous
        )

        return evaluations, summary
