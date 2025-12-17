import React from 'react';
import { Bot, Info } from 'lucide-react';

const Header: React.FC = () => {
  return (
    <header className="bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between sticky top-0 z-10 shadow-sm">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-emerald-100 rounded-lg">
          <Bot className="w-6 h-6 text-emerald-600" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-slate-800">Bangla FAQ Assistant</h1>
          <p className="text-xs text-slate-500">আপনার ব্যক্তিগত এআই সহযোগী</p>
        </div>
      </div>
      <button className="p-2 hover:bg-slate-100 rounded-full transition-colors text-slate-400 hover:text-emerald-600">
        <Info className="w-5 h-5" />
      </button>
    </header>
  );
};

export default Header;
