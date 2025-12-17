import React, { useState } from 'react';
import { Message, MessageSender } from '../types';
import { Bot, User, Play, Volume2, Loader2, Tag } from 'lucide-react';
import { ApiService } from '../services/api';
import { hexToAudioUrl } from '../utils/audio';

interface MessageBubbleProps {
  message: Message;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isUser = message.sender === MessageSender.User;
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoadingAudio, setIsLoadingAudio] = useState(false);
  const [audioEl, setAudioEl] = useState<HTMLAudioElement | null>(null);

  const handlePlayTTS = async () => {
    if (isPlaying && audioEl) {
      audioEl.pause();
      audioEl.currentTime = 0;
      setIsPlaying(false);
      return;
    }

    setIsLoadingAudio(true);
    try {
      // 1. Fetch hex string
      const hexAudio = await ApiService.getTTS(message.text);
      
      // 2. Convert to Blob URL
      const audioUrl = hexToAudioUrl(hexAudio);
      
      if (audioUrl) {
        const audio = new Audio(audioUrl);
        setAudioEl(audio);
        
        audio.onended = () => {
            setIsPlaying(false);
            URL.revokeObjectURL(audioUrl); // Cleanup
        };
        
        audio.onplay = () => setIsPlaying(true);
        
        await audio.play();
      }
    } catch (err) {
      console.error("Failed to play audio", err);
    } finally {
      setIsLoadingAudio(false);
    }
  };

  return (
    <div className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'} mb-6 group`}>
      <div className={`flex max-w-[85%] md:max-w-[70%] gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
        
        {/* Avatar */}
        <div className={`flex-shrink-0 w-8 h-8 md:w-10 md:h-10 rounded-full flex items-center justify-center shadow-sm ${
          isUser ? 'bg-indigo-100' : 'bg-emerald-100'
        }`}>
          {isUser ? <User size={18} className="text-indigo-600" /> : <Bot size={18} className="text-emerald-600" />}
        </div>

        {/* Bubble */}
        <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
          <div className={`px-5 py-3 rounded-2xl shadow-sm text-sm md:text-base leading-relaxed ${
            isUser 
              ? 'bg-indigo-600 text-white rounded-tr-none' 
              : 'bg-white text-slate-700 border border-slate-100 rounded-tl-none'
          }`}>
            {message.text}
          </div>

          {/* Meta Data & Actions */}
          <div className="flex items-center gap-2 mt-2 px-1">
             <span className="text-[10px] text-slate-400 uppercase font-medium">
               {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
             </span>

             {!isUser && message.category && (
                <span className="flex items-center gap-1 text-[10px] bg-slate-100 text-slate-500 px-2 py-0.5 rounded-full">
                    <Tag size={10} />
                    {message.category}
                </span>
             )}

             {!isUser && (
               <button 
                onClick={handlePlayTTS}
                disabled={isLoadingAudio}
                className="flex items-center gap-1 text-[10px] font-medium text-emerald-600 hover:text-emerald-700 hover:bg-emerald-50 px-2 py-0.5 rounded-full transition-colors"
               >
                 {isLoadingAudio ? (
                   <Loader2 size={12} className="animate-spin" />
                 ) : isPlaying ? (
                   <Volume2 size={12} />
                 ) : (
                   <Play size={12} />
                 )}
                 {isPlaying ? 'বন্ধ করুন' : 'শুনুন'}
               </button>
             )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MessageBubble;
