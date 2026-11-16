import axios from 'axios';
import type { AnalysisResponse, ComparisonResponse, ModelType } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

export const analyzeImage = async (
  file: File,
  modelType: ModelType = 'yolov8s',
  confPpe: number = 0.25,
  confPerson: number = 0.35,
  iouThreshold: number = 0.45
): Promise<AnalysisResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('model_type', modelType);
  formData.append('conf_ppe', confPpe.toString());
  formData.append('conf_person', confPerson.toString());
  formData.append('iou_threshold', iouThreshold.toString());

  const response = await apiClient.post<AnalysisResponse>('/api/analyze', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

export const compareModels = async (
  file: File,
  confPpe: number = 0.25,
  confPerson: number = 0.35,
  iouThreshold: number = 0.45
): Promise<ComparisonResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('conf_ppe', confPpe.toString());
  formData.append('conf_person', confPerson.toString());
  formData.append('iou_threshold', iouThreshold.toString());

  const response = await apiClient.post<ComparisonResponse>('/api/compare', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};
