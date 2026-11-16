import React from 'react';
import { AlertTriangle, ShieldCheck, UserX, Info } from 'lucide-react';
import type { SummaryMetrics } from '../types';

interface SceneAlertBannerProps {
  summary: SummaryMetrics | null;
}

export const SceneAlertBanner: React.FC<SceneAlertBannerProps> = ({ summary }) => {
  if (!summary) return null;

  if (summary.is_scene_dangerous) {
    return (
      <div className="rounded-xl p-4 shadow-sm border bg-rose-50 border-rose-300 text-rose-900 flex items-start gap-3.5 transition">
        <div className="w-9 h-9 rounded-full bg-rose-600 text-white flex items-center justify-center shrink-0 shadow">
          <AlertTriangle className="w-5 h-5" />
        </div>
        <div>
          <h3 className="font-bold text-sm tracking-tight">
            🚨 DANGER ALERT: {summary.danger_no_helmet} WORKER(S) NOT WEARING SAFETY HELMET!
          </h3>
          <p className="text-xs mt-0.5 opacity-90">
            Critical workplace safety violation detected. Immediate on-site inspection and corrective action required.
          </p>
        </div>
      </div>
    );
  }

  if (summary.total_workers > 0 && summary.helmet_compliance_rate === 100.0) {
    return (
      <div className="rounded-xl p-4 shadow-sm border bg-emerald-50 border-emerald-300 text-emerald-900 flex items-start gap-3.5 transition">
        <div className="w-9 h-9 rounded-full bg-emerald-600 text-white flex items-center justify-center shrink-0 shadow">
          <ShieldCheck className="w-5 h-5" />
        </div>
        <div>
          <h3 className="font-bold text-sm tracking-tight">
            ✅ SAFE ENVIRONMENT: 100% WORKERS COMPLIANT WITH HELMET REGULATIONS
          </h3>
          <p className="text-xs mt-0.5 opacity-90">
            All recorded workers are properly equipped with safety helmets in their assigned zones.
          </p>
        </div>
      </div>
    );
  }

  if (summary.total_workers === 0) {
    return (
      <div className="rounded-xl p-4 shadow-sm border bg-slate-100 border-slate-300 text-slate-800 flex items-start gap-3.5 transition">
        <div className="w-9 h-9 rounded-full bg-slate-400 text-white flex items-center justify-center shrink-0">
          <UserX className="w-5 h-5" />
        </div>
        <div>
          <h3 className="font-bold text-sm tracking-tight">
            ℹ️ NO WORKERS DETECTED IN FRAME
          </h3>
          <p className="text-xs mt-0.5 opacity-90">
            No person or safety vest detected matching the established confidence thresholds.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl p-4 shadow-sm border bg-amber-50 border-amber-300 text-amber-900 flex items-start gap-3.5 transition">
      <div className="w-9 h-9 rounded-full bg-amber-500 text-white flex items-center justify-center shrink-0 shadow">
        <Info className="w-5 h-5" />
      </div>
      <div>
        <h3 className="font-bold text-sm tracking-tight">
          ⚠️ ATTENTION: POTENTIAL ISSUES OR INSUFFICIENT EVIDENCE
        </h3>
        <p className="text-xs mt-0.5 opacity-90">
          Identified {summary.insufficient_evidence} instance(s) with occluded head region or missing safety vest.
        </p>
      </div>
    </div>
  );
};
