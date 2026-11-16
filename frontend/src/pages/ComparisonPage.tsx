import React, { useState } from 'react';
import { Cpu, Zap, Table, Lightbulb, Gauge, Star, History } from 'lucide-react';
import { ImageDropzone } from '../components/ImageDropzone';
import { compareModels } from '../services/api';
import type { ComparisonResponse } from '../types';

export const ComparisonPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [comparisonData, setComparisonData] = useState<ComparisonResponse | null>(null);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setComparisonData(null);
  };

  const handleClear = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(null);
    setPreviewUrl(null);
    setComparisonData(null);
  };

  const handleRunBenchmark = async () => {
    if (!selectedFile) return;
    setIsLoading(true);

    try {
      const data = await compareModels(selectedFile, 0.25, 0.35, 0.45);
      setComparisonData(data);
    } catch (err: any) {
      console.error(err);
      alert(err.response?.data?.detail || err.message || 'Benchmark comparison failed.');
    } finally {
      setIsLoading(false);
    }
  };

  const v8n = comparisonData?.results['yolov8n'];
  const v8s = comparisonData?.results['yolov8s'];
  const v5 = comparisonData?.results['yolov5'];

  return (
    <div className="flex flex-col gap-6">
      {/* Top Banner & Control Section */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 pb-4 border-b border-slate-100">
          <div>
            <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <Cpu className="w-5 h-5 text-teal-600" />
              Side-by-Side Model Benchmark Studio
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Upload a construction-site image to execute <strong>YOLOv8n (Nano)</strong>,{' '}
              <strong>YOLOv8s (Fine-Tuned)</strong>, and <strong>YOLOv5 (Classic)</strong> in parallel under identical Rule Engine conditions.
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs font-mono bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-lg text-slate-600">
            <span className="w-2 h-2 rounded-full bg-teal-500"></span>
            <span>Fixed Params: PPE=0.25 | Person=0.35 | IoU=0.45</span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-12 gap-5 mt-4 items-center">
          {/* Dropzone */}
          <div className="md:col-span-8">
            <ImageDropzone
              selectedFile={selectedFile}
              previewUrl={previewUrl}
              onFileSelect={handleFileSelect}
              onClear={handleClear}
              accentColor="teal"
              label="Select Test Image for Multi-Model Benchmark"
            />
          </div>

          {/* Action Button */}
          <div className="md:col-span-4 flex flex-col gap-2">
            <button
              onClick={handleRunBenchmark}
              disabled={!selectedFile || isLoading}
              className="w-full py-4 px-4 rounded-xl bg-teal-600 hover:bg-teal-700 disabled:bg-slate-300 disabled:cursor-not-allowed text-white font-bold text-sm shadow-md transition flex flex-col items-center justify-center gap-1.5 cursor-pointer"
            >
              <div className="flex items-center gap-2">
                {isLoading ? (
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                ) : (
                  <Zap className="w-5 h-5 text-amber-300 fill-amber-300" />
                )}
                <span>{isLoading ? 'Running 3-Model Benchmark...' : 'Run 3-Model Benchmark'}</span>
              </div>
              <span className="text-[10px] font-normal text-teal-100">
                Executes v8n vs v8s vs v5 in parallel
              </span>
            </button>
            <p className="text-[11px] text-slate-400 text-center">
              Measures exact latency, detection recall, and compliance score
            </p>
          </div>
        </div>
      </div>

      {/* Benchmark Results */}
      {comparisonData && v8n && v8s && v5 && (
        <div className="flex flex-col gap-6">
          {/* 1. Quantitative Performance Matrix */}
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100 mb-4">
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <Table className="w-4 h-4 text-teal-600" />
                Quantitative Performance Matrix
              </h3>
              <span className="text-xs text-slate-500 font-mono">Test Platform: Real-Time Host GPU/CPU</span>
            </div>

            <div className="overflow-x-auto custom-scrollbar">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="bg-slate-50 border-y border-slate-200 text-slate-600 uppercase tracking-wider">
                    <th className="py-3 px-4 font-semibold">Evaluation Metric</th>
                    <th className="py-3 px-4 font-bold text-blue-700 bg-blue-50/50 border-x border-slate-200">
                      <div className="flex items-center justify-between">
                        <span>YOLOv8n (Nano)</span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-200 text-blue-800">Ultra-Fast</span>
                      </div>
                    </th>
                    <th className="py-3 px-4 font-bold text-emerald-800 bg-emerald-50/60 border-r border-slate-200">
                      <div className="flex items-center justify-between">
                        <span>YOLOv8s (Small)</span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-200 text-emerald-900">Fine-Tuned ⭐</span>
                      </div>
                    </th>
                    <th className="py-3 px-4 font-bold text-slate-700 bg-slate-100/50">
                      <div className="flex items-center justify-between">
                        <span>YOLOv5 (Classic)</span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-200 text-slate-700">Baseline</span>
                      </div>
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-medium">
                  <tr className="hover:bg-slate-50/80 transition">
                    <td className="py-2.5 px-4 text-slate-800">Inference Latency (ms)</td>
                    <td className="py-2.5 px-4 font-mono bg-blue-50/30 border-x border-slate-100">
                      <span className="text-blue-600 font-bold">{v8n.inference_time_ms} ms</span>{' '}
                      <span className="text-[9px] bg-blue-100 text-blue-700 px-1 py-0.5 rounded">Fastest ⚡</span>
                    </td>
                    <td className="py-2.5 px-4 font-mono bg-emerald-50/30 border-r border-slate-100">
                      <span className="text-emerald-700 font-bold">{v8s.inference_time_ms} ms</span>
                    </td>
                    <td className="py-2.5 px-4 font-mono bg-slate-50/30">
                      <span className="text-slate-700 font-bold">{v5.inference_time_ms} ms</span>
                    </td>
                  </tr>

                  <tr className="hover:bg-slate-50/80 transition">
                    <td className="py-2.5 px-4 text-slate-800">Total Workers Detected</td>
                    <td className="py-2.5 px-4 font-mono bg-blue-50/30 border-x border-slate-100">{v8n.summary.total_workers} persons</td>
                    <td className="py-2.5 px-4 font-mono bg-emerald-50/30 border-r border-slate-100">
                      <strong className="text-emerald-700 font-black">{v8s.summary.total_workers} persons</strong>
                    </td>
                    <td className="py-2.5 px-4 font-mono bg-slate-50/30">{v5.summary.total_workers} persons</td>
                  </tr>

                  <tr className="hover:bg-slate-50/80 transition">
                    <td className="py-2.5 px-4 text-slate-800">Helmet Compliance Rate (%)</td>
                    <td className="py-2.5 px-4 font-mono bg-blue-50/30 border-x border-slate-100">{v8n.summary.helmet_compliance_rate.toFixed(1)}%</td>
                    <td className="py-2.5 px-4 font-mono bg-emerald-50/30 border-r border-slate-100">
                      <strong className="text-emerald-700">{v8s.summary.helmet_compliance_rate.toFixed(1)}%</strong>
                    </td>
                    <td className="py-2.5 px-4 font-mono bg-slate-50/30">{v5.summary.helmet_compliance_rate.toFixed(1)}%</td>
                  </tr>

                  <tr className="hover:bg-slate-50/80 transition">
                    <td className="py-2.5 px-4 text-slate-800">High-Risk Violations Count</td>
                    <td className="py-2.5 px-4 font-mono bg-blue-50/30 border-x border-slate-100">
                      {v8n.summary.danger_no_helmet > 0 ? (
                        <span className="text-rose-600 font-bold">{v8n.summary.danger_no_helmet}</span>
                      ) : (
                        '0'
                      )}
                    </td>
                    <td className="py-2.5 px-4 font-mono bg-emerald-50/30 border-r border-slate-100">
                      {v8s.summary.danger_no_helmet > 0 ? (
                        <span className="text-rose-600 font-bold">{v8s.summary.danger_no_helmet}</span>
                      ) : (
                        '0'
                      )}
                    </td>
                    <td className="py-2.5 px-4 font-mono bg-slate-50/30">
                      {v5.summary.danger_no_helmet > 0 ? (
                        <span className="text-rose-600 font-bold">{v5.summary.danger_no_helmet}</span>
                      ) : (
                        '0'
                      )}
                    </td>
                  </tr>

                  <tr className="hover:bg-slate-50/80 transition">
                    <td className="py-2.5 px-4 text-slate-800">Uncertain / Clipped Cases</td>
                    <td className="py-2.5 px-4 font-mono bg-blue-50/30 border-x border-slate-100">{v8n.summary.insufficient_evidence}</td>
                    <td className="py-2.5 px-4 font-mono bg-emerald-50/30 border-r border-slate-100">{v8s.summary.insufficient_evidence}</td>
                    <td className="py-2.5 px-4 font-mono bg-slate-50/30">{v5.summary.insufficient_evidence}</td>
                  </tr>

                  <tr className="hover:bg-slate-50/80 transition">
                    <td className="py-2.5 px-4 text-slate-800">Overall Site Risk Verdict</td>
                    <td className="py-2.5 px-4 font-mono bg-blue-50/30 border-x border-slate-100">
                      {v8n.summary.is_scene_dangerous ? (
                        <span className="text-rose-600 font-bold">⚠️ DANGER</span>
                      ) : (
                        <span className="text-emerald-600 font-bold">✅ SAFE</span>
                      )}
                    </td>
                    <td className="py-2.5 px-4 font-mono bg-emerald-50/30 border-r border-slate-100">
                      {v8s.summary.is_scene_dangerous ? (
                        <span className="text-rose-600 font-bold">⚠️ DANGER</span>
                      ) : (
                        <span className="text-emerald-600 font-bold">✅ SAFE</span>
                      )}
                    </td>
                    <td className="py-2.5 px-4 font-mono bg-slate-50/30">
                      {v5.summary.is_scene_dangerous ? (
                        <span className="text-rose-600 font-bold">⚠️ DANGER</span>
                      ) : (
                        <span className="text-emerald-600 font-bold">✅ SAFE</span>
                      )}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* 2. Side-by-Side Visual Comparison Cards */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
            {/* Card 1: YOLOv8n */}
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-4 flex flex-col">
              <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-100">
                <div>
                  <span className="text-xs font-bold text-blue-700">YOLOv8n (Nano)</span>
                  <span className="block text-[10px] text-slate-400">Lightweight Architecture</span>
                </div>
                <span className="text-xs font-mono font-bold bg-blue-50 text-blue-700 border border-blue-200 px-2 py-0.5 rounded">
                  {v8n.inference_time_ms} ms
                </span>
              </div>
              <div className="relative bg-slate-900/5 rounded-lg border border-slate-200 min-h-[260px] flex items-center justify-center overflow-hidden mb-3">
                <img src={v8n.image_base64} alt="YOLOv8n Result" className="w-full h-auto max-h-[320px] object-contain" />
              </div>
              <div className="text-xs space-y-1 bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                <div className="flex justify-between"><span>Workers:</span><strong>{v8n.summary.total_workers}</strong></div>
                <div className="flex justify-between"><span>Compliance:</span><strong className="text-emerald-600">{v8n.summary.helmet_compliance_rate.toFixed(1)}%</strong></div>
                <div className="flex justify-between"><span>Violations:</span><strong className="text-rose-600">{v8n.summary.danger_no_helmet}</strong></div>
              </div>
            </div>

            {/* Card 2: YOLOv8s */}
            <div className="bg-white rounded-xl shadow-sm border-2 border-emerald-500 p-4 flex flex-col relative">
              <div className="absolute -top-3 right-4 bg-emerald-600 text-white text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full shadow">
                Recommended Model
              </div>
              <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-100">
                <div>
                  <span className="text-xs font-bold text-emerald-800">YOLOv8s (Small)</span>
                  <span className="block text-[10px] text-slate-400">Trained on Custom PPE Weights</span>
                </div>
                <span className="text-xs font-mono font-bold bg-emerald-50 text-emerald-800 border border-emerald-200 px-2 py-0.5 rounded">
                  {v8s.inference_time_ms} ms
                </span>
              </div>
              <div className="relative bg-slate-900/5 rounded-lg border border-slate-200 min-h-[260px] flex items-center justify-center overflow-hidden mb-3">
                <img src={v8s.image_base64} alt="YOLOv8s Result" className="w-full h-auto max-h-[320px] object-contain" />
              </div>
              <div className="text-xs space-y-1 bg-emerald-50/50 p-2.5 rounded-lg border border-emerald-200">
                <div className="flex justify-between"><span>Workers:</span><strong>{v8s.summary.total_workers}</strong></div>
                <div className="flex justify-between"><span>Compliance:</span><strong className="text-emerald-600">{v8s.summary.helmet_compliance_rate.toFixed(1)}%</strong></div>
                <div className="flex justify-between"><span>Violations:</span><strong className="text-rose-600">{v8s.summary.danger_no_helmet}</strong></div>
              </div>
            </div>

            {/* Card 3: YOLOv5 */}
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-4 flex flex-col">
              <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-100">
                <div>
                  <span className="text-xs font-bold text-slate-800">YOLOv5 (Classic)</span>
                  <span className="block text-[10px] text-slate-400">Standard Baseline</span>
                </div>
                <span className="text-xs font-mono font-bold bg-slate-100 text-slate-800 border border-slate-200 px-2 py-0.5 rounded">
                  {v5.inference_time_ms} ms
                </span>
              </div>
              <div className="relative bg-slate-900/5 rounded-lg border border-slate-200 min-h-[260px] flex items-center justify-center overflow-hidden mb-3">
                <img src={v5.image_base64} alt="YOLOv5 Result" className="w-full h-auto max-h-[320px] object-contain" />
              </div>
              <div className="text-xs space-y-1 bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                <div className="flex justify-between"><span>Workers:</span><strong>{v5.summary.total_workers}</strong></div>
                <div className="flex justify-between"><span>Compliance:</span><strong className="text-emerald-600">{v5.summary.helmet_compliance_rate.toFixed(1)}%</strong></div>
                <div className="flex justify-between"><span>Violations:</span><strong className="text-rose-600">{v5.summary.danger_no_helmet}</strong></div>
              </div>
            </div>
          </div>

          {/* 3. Engineering Recommendations */}
          <div className="bg-gradient-to-br from-slate-900 to-slate-800 text-white rounded-xl p-5 shadow-sm">
            <h4 className="text-sm font-bold flex items-center gap-2 mb-3 text-teal-400">
              <Lightbulb className="w-4 h-4" />
              Engineering Deployment Analysis & Practical Takeaways
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-slate-300 leading-relaxed">
              <div className="bg-slate-800/80 p-3.5 rounded-lg border border-slate-700">
                <h5 className="font-bold text-blue-400 mb-1 flex items-center gap-1.5">
                  <Gauge className="w-3.5 h-3.5" /> YOLOv8n (Edge Camera)
                </h5>
                <p>
                  Best for resource-constrained embedded systems (NVIDIA Jetson Nano, Raspberry Pi 5). Offers highest frame rate but may miss tiny helmets at long camera distances.
                </p>
              </div>
              <div className="bg-slate-800/80 p-3.5 rounded-lg border border-emerald-500/80">
                <h5 className="font-bold text-emerald-400 mb-1 flex items-center gap-1.5">
                  <Star className="w-3.5 h-3.5" /> YOLOv8s (Server AI - Best)
                </h5>
                <p>
                  Optimal trade-off for on-site centralized servers or cloud GPU inference. Demonstrates the highest recall on small objects (helmets, goggles) and tight crowds.
                </p>
              </div>
              <div className="bg-slate-800/80 p-3.5 rounded-lg border border-slate-700">
                <h5 className="font-bold text-slate-400 mb-1 flex items-center gap-1.5">
                  <History className="w-3.5 h-3.5" /> YOLOv5 (Classic Baseline)
                </h5>
                <p>
                  Extremely mature and proven in legacy industrial deployments. Serves as a reliable baseline, though YOLOv8 exhibits superior feature pyramid aggregation on PPE.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
