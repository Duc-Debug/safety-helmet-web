import React, { useState } from 'react';
import { Brain, Sliders, BookmarkCheck } from 'lucide-react';
import { ImageDropzone } from '../components/ImageDropzone';
import { ModelSelector } from '../components/ModelSelector';
import { MetricsGrid } from '../components/MetricsGrid';
import { SceneAlertBanner } from '../components/SceneAlertBanner';
import { AnnotatedViewer } from '../components/AnnotatedViewer';
import { RuleTraceTable } from '../components/RuleTraceTable';
import { analyzeImage } from '../services/api';
import type { AnalysisResponse, ModelType } from '../types';

interface DashboardPageProps {
  onNavigateCompare: () => void;
  onNavigateTuning: () => void;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({
  onNavigateCompare,
  onNavigateTuning,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelType>('yolov8s');
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

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setIsLoading(true);

    try {
      const data = await analyzeImage(selectedFile, selectedModel, 0.25, 0.35, 0.45);
      setResult(data);
    } catch (err: any) {
      console.error(err);
      alert(err.response?.data?.detail || err.message || 'Failed to analyze image.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      {/* 2 Columns Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Controls */}
        <div className="lg:col-span-5 flex flex-col gap-5">
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
            <h2 className="text-base font-bold text-slate-900 flex items-center gap-2 mb-4 pb-2 border-b border-slate-100">
              <Sliders className="w-4 h-4 text-blue-600" />
              Analysis Control Panel
            </h2>

            {/* Image Dropzone */}
            <div className="mb-4">
              <ImageDropzone
                selectedFile={selectedFile}
                previewUrl={previewUrl}
                onFileSelect={handleFileSelect}
                onClear={handleClear}
              />
            </div>

            {/* Compact Model Selector */}
            <div className="mb-5">
              <ModelSelector
                selectedModel={selectedModel}
                onChange={setSelectedModel}
                onNavigateCompare={onNavigateCompare}
                onNavigateTuning={onNavigateTuning}
              />
            </div>

            {/* Analyze Action Button */}
            <button
              onClick={handleAnalyze}
              disabled={!selectedFile || isLoading}
              className="w-full py-3 px-4 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 disabled:cursor-not-allowed text-white font-semibold text-sm shadow-md transition flex items-center justify-center gap-2 cursor-pointer"
            >
              {isLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  <span>Detecting & Reasoning...</span>
                </>
              ) : (
                <>
                  <Brain className="w-4 h-4" />
                  <span>Analyze & Reason Safety Rules</span>
                </>
              )}
            </button>
          </div>

          {/* Rule Base Reference Box */}
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-4 text-xs text-slate-600">
            <h3 className="font-bold text-slate-800 uppercase mb-2 text-[11px] flex items-center gap-1.5">
              <BookmarkCheck className="w-3.5 h-3.5 text-slate-500" />
              Rule Base Reference
            </h3>
            <ul className="space-y-1.5">
              <li className="flex items-center justify-between">
                <span className="font-mono font-bold text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200">
                  R01
                </span>
                <span className="text-slate-600">Full Compliance: Helmet + Safety Vest (Safe)</span>
              </li>
              <li className="flex items-center justify-between">
                <span className="font-mono font-bold text-amber-700 bg-amber-50 px-1.5 py-0.5 rounded border border-amber-200">
                  R02
                </span>
                <span className="text-slate-600">Helmet equipped, safety vest missing (Warning)</span>
              </li>
              <li className="flex items-center justify-between">
                <span className="font-mono font-bold text-rose-700 bg-rose-50 px-1.5 py-0.5 rounded border border-rose-200">
                  R03
                </span>
                <span className="text-slate-600">Clear head detected but NO HELMET (High Risk)</span>
              </li>
              <li className="flex items-center justify-between">
                <span className="font-mono font-bold text-rose-700 bg-rose-50 px-1.5 py-0.5 rounded border border-rose-200">
                  R04
                </span>
                <span className="text-slate-600">Inferred from vest: NO HELMET (High Risk)</span>
              </li>
              <li className="flex items-center justify-between">
                <span className="font-mono font-bold text-slate-600 bg-slate-100 px-1.5 py-0.5 rounded border border-slate-200">
                  R05
                </span>
                <span className="text-slate-600">Head occluded/clipped at border (Uncertain)</span>
              </li>
            </ul>
          </div>
        </div>

        {/* Right Column: Visuals & Metrics */}
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

      {/* Bottom Section: Rule Trace Table */}
      <RuleTraceTable evaluations={result ? result.evaluations : []} />
    </div>
  );
};
