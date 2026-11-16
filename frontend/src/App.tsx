import { useState } from 'react';
import { Navbar } from './components/Navbar';
import { DashboardPage } from './pages/DashboardPage';
import { ComparisonPage } from './pages/ComparisonPage';
import { TuningPage } from './pages/TuningPage';
import { ShieldCheck, Cpu, Code2 } from 'lucide-react';

export function App() {
  const [currentTab, setCurrentTab] = useState<'dashboard' | 'compare' | 'tuning'>('dashboard');

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-800">
      <Navbar currentTab={currentTab} onTabChange={setCurrentTab} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {currentTab === 'dashboard' && (
          <DashboardPage
            onNavigateCompare={() => setCurrentTab('compare')}
            onNavigateTuning={() => setCurrentTab('tuning')}
          />
        )}

        {currentTab === 'compare' && <ComparisonPage />}

        {currentTab === 'tuning' && <TuningPage />}
      </main>

      <footer className="bg-white border-t border-slate-200 mt-auto py-5">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500 gap-3">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-600" />
            <span className="font-semibold text-slate-700">
              Safety Vision AI & Rule-Based Reasoning Engine
            </span>
            <span className="text-slate-300">|</span>
            <span>R01–R05 Forward Chaining</span>
          </div>
          <div className="flex items-center gap-4 text-[11px]">
            <span className="flex items-center gap-1">
              <Cpu className="w-3.5 h-3.5 text-blue-500" />
              YOLOv8s / YOLOv8n / YOLOv5
            </span>
            <span className="flex items-center gap-1">
              <Code2 className="w-3.5 h-3.5 text-indigo-500" />
              React + Vite + Tailwind v4
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
