import React from 'react';
import { Image as ImageIcon, Download, CheckCircle2 } from 'lucide-react';

interface AnnotatedViewerProps {
  imageBase64: string | null;
  modelUsed?: string;
  inferenceTimeMs?: number;
  isLoading?: boolean;
}

export const AnnotatedViewer: React.FC<AnnotatedViewerProps> = ({
  imageBase64,
  modelUsed,
  inferenceTimeMs,
  isLoading,
}) => {
  const handleDownload = () => {
    if (!imageBase64) return;
    const link = document.createElement('a');
    link.href = imageBase64;
    link.download = `safety_annotated_${Date.now()}.jpg`;
    link.click();
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-4 flex-1 flex flex-col">
      <div className="flex items-center justify-between pb-3 border-b border-slate-100 mb-3">
        <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
          <ImageIcon className="w-4 h-4 text-blue-600" />
          Visualized Result (Bounding Boxes & Status)
        </h3>

        <div className="flex items-center gap-2">
          {imageBase64 && (
            <button
              onClick={handleDownload}
              className="text-[11px] text-slate-600 hover:text-slate-900 font-semibold px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 border border-slate-200 transition flex items-center gap-1 cursor-pointer"
              title="Download annotated image"
            >
              <Download className="w-3 h-3" />
              Download
            </button>
          )}

          {imageBase64 ? (
            <span className="text-[11px] text-emerald-600 font-semibold flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5" />
              {modelUsed?.toUpperCase() || 'YOLO'} Done {inferenceTimeMs ? `(${inferenceTimeMs} ms)` : ''}
            </span>
          ) : (
            <span className="text-[11px] text-slate-400 italic">No data yet</span>
          )}
        </div>
      </div>

      <div className="relative bg-slate-900/5 rounded-lg border border-slate-200 flex items-center justify-center min-h-[320px] overflow-hidden flex-1">
        {isLoading ? (
          <div className="flex flex-col items-center justify-center p-8 text-slate-500">
            <div className="w-10 h-10 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mb-3"></div>
            <p className="text-sm font-semibold text-slate-700">Running AI Inference & Rule Reasoning...</p>
            <p className="text-xs text-slate-400 mt-1">Please wait a few moments</p>
          </div>
        ) : imageBase64 ? (
          <img
            src={imageBase64}
            alt="Annotated Safety Result"
            className="w-full h-auto max-h-[520px] object-contain rounded-md shadow-inner"
          />
        ) : (
          <div className="text-center p-8 text-slate-400">
            <ImageIcon className="w-12 h-12 mx-auto mb-2 text-slate-300 stroke-[1.2]" />
            <p className="text-sm font-medium">Upload an image and click "Analyze" to inspect detected bounding boxes</p>
          </div>
        )}
      </div>
    </div>
  );
};
