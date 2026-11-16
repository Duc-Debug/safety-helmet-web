import React from 'react';
import { Layers, ArrowRight } from 'lucide-react';
import type { ModelType } from '../types';

interface ModelSelectorProps {
  selectedModel: ModelType;
  onChange: (model: ModelType) => void;
  onNavigateCompare?: () => void;
  onNavigateTuning?: () => void;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  selectedModel,
  onChange,
  onNavigateCompare,
  onNavigateTuning,
}) => {
  return (
    <div className="bg-slate-50 p-3.5 rounded-xl border border-slate-200">
      <div className="flex items-center justify-between mb-2">
        <label className="text-xs font-bold text-slate-700 uppercase flex items-center gap-1.5">
          <Layers className="w-3.5 h-3.5 text-blue-600" />
          Model Architecture
        </label>
        {onNavigateCompare && (
          <button
            type="button"
            onClick={onNavigateCompare}
            className="text-[11px] text-teal-600 hover:text-teal-800 font-semibold inline-flex items-center gap-1 hover:underline cursor-pointer"
          >
            Benchmark All <ArrowRight className="w-2.5 h-2.5" />
          </button>
        )}
      </div>

      {/* 3 Model Radio Pills */}
      <div className="grid grid-cols-3 gap-1.5 mb-2.5">
        <label className="cursor-pointer">
          <input
            type="radio"
            name="model_type"
            value="yolov8s"
            checked={selectedModel === 'yolov8s'}
            onChange={() => onChange('yolov8s')}
            className="peer sr-only"
          />
          <div className="px-2 py-1.5 rounded-lg border border-slate-200 text-center peer-checked:border-blue-600 peer-checked:bg-blue-50/70 peer-checked:text-blue-700 transition hover:bg-slate-100">
            <span className="block text-xs font-bold">YOLOv8s</span>
            <span className="block text-[9px] text-emerald-600 font-semibold">Fine-Tuned ⭐</span>
          </div>
        </label>

        <label className="cursor-pointer">
          <input
            type="radio"
            name="model_type"
            value="yolov8n"
            checked={selectedModel === 'yolov8n'}
            onChange={() => onChange('yolov8n')}
            className="peer sr-only"
          />
          <div className="px-2 py-1.5 rounded-lg border border-slate-200 text-center peer-checked:border-blue-600 peer-checked:bg-blue-50/70 peer-checked:text-blue-700 transition hover:bg-slate-100">
            <span className="block text-xs font-bold">YOLOv8n</span>
            <span className="block text-[9px] text-slate-500 font-medium">Ultra-Fast</span>
          </div>
        </label>

        <label className="cursor-pointer">
          <input
            type="radio"
            name="model_type"
            value="yolov5"
            checked={selectedModel === 'yolov5'}
            onChange={() => onChange('yolov5')}
            className="peer sr-only"
          />
          <div className="px-2 py-1.5 rounded-lg border border-slate-200 text-center peer-checked:border-blue-600 peer-checked:bg-blue-50/70 peer-checked:text-blue-700 transition hover:bg-slate-100">
            <span className="block text-xs font-bold">YOLOv5</span>
            <span className="block text-[9px] text-slate-500 font-medium">Baseline</span>
          </div>
        </label>
      </div>

      {/* Compact Active Parameters Bar */}
      <div className="flex items-center justify-between text-[11px] text-slate-500 bg-white px-2.5 py-1.5 rounded-lg border border-slate-200/80">
        <span className="font-medium text-[10px]">Active Params:</span>
        <div className="flex gap-1.5 font-mono text-[10px]">
          <span className="bg-slate-100 px-1.5 py-0.5 rounded text-slate-700" title="Confidence threshold for Person">
            Person: 0.35
          </span>
          <span className="bg-slate-100 px-1.5 py-0.5 rounded text-slate-700" title="Confidence threshold for PPE">
            PPE: 0.25
          </span>
          <span className="bg-slate-100 px-1.5 py-0.5 rounded text-slate-700" title="Non-Maximum Suppression IoU threshold">
            IoU: 0.45
          </span>
        </div>
        {onNavigateTuning && (
          <button
            type="button"
            onClick={onNavigateTuning}
            className="text-indigo-600 hover:text-indigo-800 font-semibold hover:underline text-[10px] cursor-pointer"
          >
            Tune
          </button>
        )}
      </div>
    </div>
  );
};
