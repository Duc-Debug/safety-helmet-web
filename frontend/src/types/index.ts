export interface DetectedObject {
  obj_id: string;
  cls_id: number;
  cls_name: string;
  conf: number;
  bbox: [number, number, number, number];
}

export interface WorkerEvaluation {
  worker_id: string;
  person_bbox: [number, number, number, number];
  head_bbox: [number, number, number, number];
  facts: string[];
  status: 'COMPLIANT' | 'WARNING_NO_VEST' | 'DANGER_NO_HELMET' | 'INSUFFICIENT_EVIDENCE' | string;
  risk_level: 'SAFE' | 'WARNING' | 'HIGH RISK' | 'UNCERTAIN' | string;
  rule_id: string;
  explanation: string;
}

export interface SummaryMetrics {
  total_workers: number;
  valid_evaluated: number;
  helmet_ok: number;
  danger_no_helmet: number;
  insufficient_evidence: number;
  helmet_compliance_rate: number;
  danger_rate: number;
  is_scene_dangerous: boolean;
}

export interface AnalysisResponse {
  summary: SummaryMetrics;
  evaluations: WorkerEvaluation[];
  image_base64: string;
  model_used?: string;
  inference_time_ms?: number;
}

export interface ModelBenchmarkResult {
  model_key: string;
  model_title: string;
  inference_time_ms: number;
  summary: SummaryMetrics;
  evaluations: WorkerEvaluation[];
  image_base64: string;
}

export interface ComparisonResponse {
  results: Record<string, ModelBenchmarkResult>;
}

export type ModelType = 'yolov8s' | 'yolov8n' | 'yolov5';
