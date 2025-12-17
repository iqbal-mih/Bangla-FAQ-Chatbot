import React, { useState, useRef, useEffect } from 'react';
import { Send, Image as ImageIcon } from 'lucide-react';
import AudioRecorder from './AudioRecorder';

interface InputAreaProps {
  onSendText: (text: string) => void;
  onSendVoice: (blob: Blob) => void;
  isLoading: boolean;
}

const InputArea: React.FC<InputAreaProps> = ({ onSendText, onSendVoice, isLoading }) => {
  const [inputText, setInputText] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + 'px';
    }
  }, [inputText]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSend = () => {
    if (inputText.trim() && !isLoading) {
      onSendText(inputText);
      setInputText('');
      if (textareaRef.current) textareaRef.current.style.height = 'auto';
    }
  };

  return (
    <div className="bg-white border-t border-slate-200 p-4">
      <div className="max-w-4xl mx-auto flex items-end gap-3">
        
        {/* Voice Recorder Button */}
        <div className="flex-shrink-0 mb-[2px]">
           <AudioRecorder 
             onRecordingComplete={onSendVoice} 
             isProcessing={isLoading}
           />
        </div>

        {/* Text Input */}
        <div className="flex-grow relative bg-slate-50 border border-slate-200 rounded-2xl focus-within:ring-2 focus-within:ring-emerald-500/20 focus-within:border-emerald-500 transition-all">
          <textarea
            ref={textareaRef}
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="আপনার প্রশ্ন লিখুন অথবা বলুন..."
            className="w-full bg-transparent border-none focus:ring-0 px-4 py-3 min-h-[50px] max-h-[120px] resize-none text-slate-700 placeholder:text-slate-400"
            rows={1}
            disabled={isLoading}
          />
        </div>

        {/* Send Button */}
        <button
          onClick={handleSend}
          disabled={!inputText.trim() || isLoading}
          className={`flex-shrink-0 p-3 rounded-full mb-[2px] transition-all duration-200 ${
            inputText.trim() && !isLoading
              ? 'bg-emerald-600 text-white shadow-md hover:bg-emerald-700 hover:scale-105'
              : 'bg-slate-100 text-slate-300 cursor-not-allowed'
          }`}
        >
          <Send size={20} />
        </button>
      </div>
      <div className="text-center mt-2">
         <p className="text-[10px] text-slate-400">এআই ভুল করতে পারে। গুরুত্বপূর্ণ তথ্যের জন্য যাচাই করুন।</p>
      </div>
    </div>
  );
};

export default InputArea;
