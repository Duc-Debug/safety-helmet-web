import React, { useState } from 'react';
import { Sliders, RotateCcw, FlaskConical, BookmarkCheck } from 'lucide-react';
import { ImageDropzone } from '../components/ImageDropzone';
import { MetricsGrid } from '../components/MetricsGrid';
import { SceneAlertBanner } from '../components/SceneAlertBanner';
import { AnnotatedViewer } from '../components/AnnotatedViewer';
import { RuleTraceTable } from '../components/RuleTraceTable';
import { analyzeImage } from '../services/api';
import type { AnalysisResponse, ModelType } from '../types';

export const TuningPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelType>('yolov8s');
  const [confPpe, setConfPpe] = useState<number>(0.25);
  const [confPerson, setConfPerson] = useState<number>(0.35);
  const [iouThreshold, setIouThreshold] = useState<number>(0.45);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResponse | null>(null);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setResult(null);
  };

  const handleClear = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
  };

  const handleResetDefaults = () => {
    setSelectedModel('yolov8s');
    setConfPpe(0.25);
    setConfPerson(0.35);
    setIouThreshold(0.45);
  };

  const handleRunExperiment = async () => {
    if (!selectedFile) return;
    setIsLoading(true);

    try {
      const data = await analyzeImage(selectedFile, selectedModel, confPpe, confPerson, iouThreshold);
      setResult(data);
    } catch (err: any) {
      console.error(err);
      alert(err.response?.data?.detail || err.message || 'Failed to run calibrated analysis.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      {/* 2 Columns Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Calibration Controls */}
        <div className="lg:col-span-5 flex flex-col gap-5">
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
            <div className="flex items-center justify-between pb-2 mb-4 border-b border-slate-100">
              <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
                <Sliders className="w-4 h-4 text-indigo-600" />
                AI Parameter Calibration
              </h2>
              <button
                type="button"
                onClick={handleResetDefaults}
                className="text-xs text-indigo-600 hover:text-indigo-800 font-semibold flex items-center gap-1 hover:underline cursor-pointer"
                title="Restore default parameters"
              >
                <RotateCcw className="w-3 h-3" /> Reset Defaults
              </button>
            </div>

            {/* Dropzone */}
            <div className="mb-4">
              <ImageDropzone
                selectedFile={selectedFile}
                previewUrl={previewUrl}
                onFileSelect={handleFileSelect}
                onClear={handleClear}
                accentColor="indigo"
                label="Upload Test Photo for Parameter Tuning"
              />
            </div>

            {/* Model Selector in Tuning */}
            <div className="mb-4">
              <label className="block text-xs font-bold text-slate-700 uppercase mb-1.5">
                Target Model
              </label>
              <div className="grid grid-cols-3 gap-1.5">
                {(['yolov8s', 'yolov8n', 'yolov5'] as ModelType[]).map((m) => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => setSelectedModel(m)}
                    className={`py-1.5 px-2 rounded-lg border text-xs font-bold uppercase transition cursor-pointer ${
                      selectedModel === m
                        ? 'border-indigo-600 bg-indigo-50 text-indigo-700 shadow-sm'
                        : 'border-slate-200 text-slate-600 hover:bg-slate-50'
                    }`}
                  >
                    {m}
                  </button>
                ))}
              </div>
            </div>

            {/* 3 Range Sliders */}
            <div className="space-y-4 mb-5 bg-slate-50 p-4 rounded-xl border border-slate-200/90 shadow-inner">
              {/* 1. PPE Confidence */}
              <div>
                <div className="flex justify-between items-center text-xs mb-1">
                  <label htmlFor="conf-ppe" className="font-bold text-slate-700">
                    PPE Confidence Threshold:
                  </label>
                  <span className="font-mono font-bold text-indigo-700 bg-indigo-100 px-2 py-0.5 rounded text-xs">
                    {confPpe.toFixed(2)}
                  </span>
                </div>
                <input
                  id="conf-ppe"
                  type="range"
                  min="0.05"
                  max="0.95"
                  step="0.05"
                  value={confPpe}
                  onChange={(e) => setConfPpe(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                />
                <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
                  <span>0.05 (Sensitive)</span>
                  <span>0.50</span>
                  <span>0.95 (Strict)</span>
                </div>
              </div>

              {/* 2. Person Confidence */}
              <div>
                <div className="flex justify-between items-center text-xs mb-1">
                  <label htmlFor="conf-person" className="font-bold text-slate-700">
                    Person Confidence Threshold:
                  </label>
                  <span className="font-mono font-bold text-indigo-700 bg-indigo-100 px-2 py-0.5 rounded text-xs">
                    {confPerson.toFixed(2)}
                  </span>
                </div>
                <input
                  id="conf-person"
                  type="range"
                  min="0.05"
                  max="0.95"
                  step="0.05"
                  value={confPerson}
                  onChange={(e) => setConfPerson(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                />
                <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
                  <span>0.05 (Sensitive)</span>
                  <span>0.35 (Optimal)</span>
                  <span>0.95 (Strict)</span>
                </div>
              </div>

              {/* 3. NMS IoU Threshold */}
              <div>
                <div className="flex justify-between items-center text-xs mb-1">
                  <label htmlFor="iou-thresh" className="font-bold text-slate-700">
                    NMS Overlap Threshold (IoU):
                  </label>
                  <span className="font-mono font-bold text-indigo-700 bg-indigo-100 px-2 py-0.5 rounded text-xs">
                    {iouThreshold.toFixed(2)}
                  </span>
                </div>
                <input
                  id="iou-thresh"
                  type="range"
                  min="0.10"
                  max="0.90"
                  step="0.05"
                  value={iouThreshold}
                  onChange={(e) => setIouThreshold(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                />
                <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
                  <span>0.10 (Aggressive)</span>
                  <span>0.45 (Standard)</span>
                  <span>0.90 (Permissive)</span>
                </div>
              </div>
            </div>

            {/* Run Button */}
            <button
              onClick={handleRunExperiment}
              disabled={!selectedFile || isLoading}
              className="w-full py-3 px-4 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-300 disabled:cursor-not-allowed text-white font-semibold text-sm shadow-md transition flex items-center justify-center gap-2 cursor-pointer"
            >
              {isLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  <span>Running Calibrated Inference...</span>
                </>
              ) : (
                <>
                  <FlaskConical className="w-4 h-4" />
                  <span>Run Parameter Experiment</span>
                </>
              )}
            </button>
          </div>

          {/* Quick Info Box */}
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-4 text-xs text-slate-600">
            <h3 className="font-bold text-slate-800 uppercase mb-2 text-[11px] flex items-center gap-1.5">
              <BookmarkCheck className="w-3.5 h-3.5 text-slate-500" />
              Calibration Advice
            </h3>
            <p className="leading-relaxed text-slate-500">
              Lowering the <strong>Person threshold</strong> below 0.25 may trigger multiple duplicate boxes on a single worker if NMS is too permissive. Keep IoU between 0.40 and 0.50 for optimal non-redundant bounding boxes.
            </p>
          </div>
        </div>

        {/* Right Column */}
        <div className="lg:col-span-7 flex flex-col gap-5">
          <SceneAlertBanner summary={result ? result.summary : null} />
          <MetricsGrid summary={result ? result.summary : null} />
          <AnnotatedViewer
            imageBase64={result ? result.image_base64 : null}
            modelUsed={result ? result.model_used : undefined}
            inferenceTimeMs={result ? result.inference_time_ms : undefined}
            isLoading={isLoading}
          />
        </div>
      </div>

      {/* Rule Trace Table */}
      <RuleTraceTable evaluations={result ? result.evaluations : []} />
    </div>
  );
};
