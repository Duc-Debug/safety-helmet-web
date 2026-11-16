import React from 'react';
import { GitBranch, CheckCircle2, AlertTriangle, XCircle, HelpCircle } from 'lucide-react';
import type { WorkerEvaluation } from '../types';

interface RuleTraceTableProps {
  evaluations: WorkerEvaluation[];
}

export const RuleTraceTable: React.FC<RuleTraceTableProps> = ({ evaluations }) => {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5 mt-2">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 pb-3 border-b border-slate-100 mb-4">
        <div>
          <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <GitBranch className="w-4 h-4 text-blue-600" />
            Rule Trace & Explainability Table
          </h2>
          <p className="text-xs text-slate-500">
            Transparent geometric association and deductive fact chains leading to each conclusion
          </p>
        </div>
        <div className="text-xs bg-slate-100 text-slate-700 px-3 py-1 rounded-full font-semibold border border-slate-200">
          {evaluations.length} entities
        </div>
      </div>

      <div className="overflow-x-auto custom-scrollbar">
        <table className="w-full text-left border-collapse text-xs">
          <thead>
            <tr className="bg-slate-50 border-y border-slate-200 text-slate-600 font-semibold uppercase tracking-wider">
              <th className="py-3 px-3">Entity</th>
              <th className="py-3 px-3">Status</th>
              <th className="py-3 px-3">Risk Level</th>
              <th className="py-3 px-3">Rule ID</th>
              <th className="py-3 px-3 min-w-[200px]">Fact Base</th>
              <th className="py-3 px-3 min-w-[280px]">Reasoning Explanation</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 text-slate-700">
            {evaluations.length === 0 ? (
              <tr>
                <td colSpan={6} className="py-8 text-center text-slate-400 italic">
                  No analysis data. Please upload an image and run reasoning.
                </td>
              </tr>
            ) : (
              evaluations.map((ev) => {
                // Status Badge
                let statusBadgeClass = 'bg-slate-200 text-slate-800 border-slate-300';
                if (ev.status === 'COMPLIANT') {
                  statusBadgeClass = 'bg-emerald-100 text-emerald-800 border-emerald-200';
                } else if (ev.status === 'WARNING_NO_VEST') {
                  statusBadgeClass = 'bg-amber-100 text-amber-800 border-amber-200';
                } else if (ev.status === 'DANGER_NO_HELMET') {
                  statusBadgeClass = 'bg-rose-100 text-rose-800 border-rose-200';
                }

                // Risk Level
                let riskIcon = <HelpCircle className="w-3.5 h-3.5 mr-1 inline" />;
                let riskTextClass = 'text-slate-500 font-medium';
                if (ev.risk_level === 'SAFE') {
                  riskIcon = <CheckCircle2 className="w-3.5 h-3.5 mr-1 inline text-emerald-600" />;
                  riskTextClass = 'text-emerald-700 font-semibold';
                } else if (ev.risk_level === 'WARNING') {
                  riskIcon = <AlertTriangle className="w-3.5 h-3.5 mr-1 inline text-amber-600" />;
                  riskTextClass = 'text-amber-700 font-semibold';
                } else if (ev.risk_level === 'HIGH RISK') {
                  riskIcon = <XCircle className="w-3.5 h-3.5 mr-1 inline text-rose-600" />;
                  riskTextClass = 'text-rose-700 font-bold';
                }

                return (
                  <tr key={ev.worker_id} className="hover:bg-slate-50 transition border-b border-slate-100">
                    <td className="py-2.5 px-3 font-mono font-bold text-slate-900">{ev.worker_id}</td>
                    <td className="py-2.5 px-3">
                      <span className={`px-2 py-0.5 rounded-full font-bold border text-[11px] ${statusBadgeClass}`}>
                        {ev.status}
                      </span>
                    </td>
                    <td className="py-2.5 px-3">
                      <span className={`flex items-center ${riskTextClass}`}>
                        {riskIcon}
                        {ev.risk_level}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 font-mono font-bold text-blue-700">{ev.rule_id}</td>
                    <td className="py-2.5 px-3">
                      <div className="flex flex-wrap gap-1">
                        {ev.facts.map((fact, idx) => {
                          let factColor = 'bg-slate-100 text-slate-700 border-slate-200';
                          if (fact === 'HelmetAssociated') factColor = 'bg-blue-100 text-blue-800 border-blue-200 font-semibold';
                          else if (fact === 'VestAssociated') factColor = 'bg-teal-100 text-teal-800 border-teal-200';
                          else if (fact === 'HeadRegionOccludedOrClipped') factColor = 'bg-amber-100 text-amber-800 border-amber-200';
                          else if (fact === 'InferredPersonFromVest') factColor = 'bg-purple-100 text-purple-800 border-purple-200';
                          return (
                            <span key={idx} className={`inline-block px-1.5 py-0.5 text-[10px] rounded border ${factColor}`}>
                              {fact}
                            </span>
                          );
                        })}
                      </div>
                    </td>
                    <td className="py-2.5 px-3 text-slate-700 leading-relaxed">{ev.explanation}</td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
