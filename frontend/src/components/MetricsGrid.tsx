import React from 'react';
import type { SummaryMetrics } from '../types';

interface MetricsGridProps {
  summary: SummaryMetrics | null;
}

export const MetricsGrid: React.FC<MetricsGridProps> = ({ summary }) => {
  return (
    <div className="grid grid-cols-3 gap-3">
      {/* Total Workers */}
      <div className="bg-white p-3.5 rounded-xl border border-slate-200 shadow-sm">
        <span className="text-[11px] font-medium text-slate-500 uppercase tracking-wider block">
          Total Workers
        </span>
        <div className="flex items-baseline gap-2 mt-1">
          <span className="text-2xl font-black text-slate-800">
            {summary ? summary.total_workers : '--'}
          </span>
          <span className="text-xs text-slate-400">persons</span>
        </div>
      </div>

      {/* Helmet Compliance Rate */}
      <div className="bg-white p-3.5 rounded-xl border border-slate-200 shadow-sm">
        <span className="text-[11px] font-medium text-emerald-600 uppercase tracking-wider block">
          Helmet Compliance
        </span>
        <div className="flex items-baseline gap-1 mt-1">
          <span className="text-2xl font-black text-emerald-600">
            {summary ? `${summary.helmet_compliance_rate.toFixed(1)}%` : '--%'}
          </span>
        </div>
      </div>

      {/* Danger Violation Rate */}
      <div className="bg-white p-3.5 rounded-xl border border-slate-200 shadow-sm">
        <span className="text-[11px] font-medium text-rose-600 uppercase tracking-wider block">
          Violation Rate
        </span>
        <div className="flex items-baseline gap-1 mt-1">
          <span className="text-2xl font-black text-rose-600">
            {summary ? `${summary.danger_rate.toFixed(1)}%` : '--%'}
          </span>
        </div>
      </div>
    </div>
  );
};
