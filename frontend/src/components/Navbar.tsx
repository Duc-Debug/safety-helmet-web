import React from 'react';
import { HardHat, GitCompare, Sliders, LayoutDashboard, Terminal } from 'lucide-react';

interface NavbarProps {
  currentTab: 'dashboard' | 'compare' | 'tuning';
  onTabChange: (tab: 'dashboard' | 'compare' | 'tuning') => void;
}

export const Navbar: React.FC<NavbarProps> = ({ currentTab, onTabChange }) => {
  return (
    <header className="bg-slate-900 text-white shadow-md border-b border-slate-800 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 py-3 sm:px-6 lg:px-8 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center space-x-3 cursor-pointer" onClick={() => onTabChange('dashboard')}>
          <div className="w-10 h-10 rounded-lg bg-amber-500 flex items-center justify-center text-slate-900 shadow">
            <HardHat className="w-6 h-6 stroke-[2.2]" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base sm:text-lg font-bold tracking-tight text-white">
                Safety Vision AI
              </h1>
              <span className="text-[10px] font-semibold uppercase px-1.5 py-0.5 bg-blue-600 rounded text-blue-100">
                Rule-Based XAI
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">
              Construction Site PPE & Safety Helmet Compliance
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center space-x-1.5 sm:space-x-2 text-xs">
          <button
            onClick={() => onTabChange('dashboard')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition ${
              currentTab === 'dashboard'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-slate-300 hover:bg-slate-800 hover:text-white'
            }`}
          >
            <LayoutDashboard className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Dashboard</span>
          </button>

          <button
            onClick={() => onTabChange('compare')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition ${
              currentTab === 'compare'
                ? 'bg-teal-600 text-white shadow-sm'
                : 'text-slate-300 hover:bg-slate-800 hover:text-white'
            }`}
          >
            <GitCompare className="w-3.5 h-3.5" />
            <span>Compare Models</span>
          </button>

          <button
            onClick={() => onTabChange('tuning')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition ${
              currentTab === 'tuning'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-300 hover:bg-slate-800 hover:text-white'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Tuning Studio</span>
          </button>

          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="hidden md:inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white border border-slate-700 transition"
            title="Swagger API documentation"
          >
            <Terminal className="w-3 h-3" />
            <span>API</span>
          </a>
        </nav>
      </div>
    </header>
  );
};
