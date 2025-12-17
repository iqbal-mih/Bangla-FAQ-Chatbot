import React, { useState, useRef, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import Header from './components/Header';
import MessageBubble from './components/MessageBubble';
import InputArea from './components/InputArea';
import { Message, MessageSender } from './types';
import { ApiService } from './services/api';
import { Sparkles } from 'lucide-react';

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      text: 'আসসালামু আলাইকুম! আমি আপনার এআই সহকারী। শিক্ষা, স্বাস্থ্য, বা প্রযুক্তি বিষয়ক যেকোনো প্রশ্ন করতে পারেন।',
      sender: MessageSender.Bot,
      timestamp: new Date(),
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const addMessage = (text: string, sender: MessageSender, category?: string) => {
    const newMessage: Message = {
      id: uuidv4(),
      text,
      sender,
      category,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, newMessage]);
  };

  const handleSendText = async (text: string) => {
    addMessage(text, MessageSender.User);
    setIsLoading(true);

    try {
      const response = await ApiService.askText(text);
      addMessage(response.answer, MessageSender.Bot, response.category);
    } catch (error) {
      addMessage("দুঃখিত, সার্ভারে সমস্যা হচ্ছে। পরে আবার চেষ্টা করুন।", MessageSender.Bot);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendVoice = async (audioBlob: Blob) => {
    // Optimistic UI updates could go here, but we wait for transcription
    setIsLoading(true);
    
    try {
      const response = await ApiService.askVoice(audioBlob);
      
      // If voice returns the recognized question, display it as User message first
      if (response.question) {
        addMessage(response.question, MessageSender.User);
      } else {
        addMessage("🎤 (Voice Query)", MessageSender.User);
      }

      // Then display bot answer
      addMessage(response.answer, MessageSender.Bot, response.category);
      
    } catch (error) {
      addMessage("দুঃখিত, আপনার কথা বুঝতে পারিনি বা সার্ভারে সমস্যা হচ্ছে।", MessageSender.Bot);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-50">
      <Header />

      {/* Main Chat Area */}
      <main className="flex-1 overflow-y-auto p-4 md:p-6 scrollbar-hide">
        <div className="max-w-4xl mx-auto flex flex-col min-h-full">
          
          {/* Messages */}
          <div className="flex-1 flex flex-col justify-end">
             {messages.length === 0 && (
               <div className="flex flex-col items-center justify-center h-64 text-slate-300">
                 <Sparkles size={48} className="mb-4 opacity-50" />
                 <p>আপনার যাত্রা শুরু করুন</p>
               </div>
             )}
             
             {messages.map((msg) => (
               <MessageBubble key={msg.id} message={msg} />
             ))}

             {isLoading && (
               <div className="flex justify-start mb-6 w-full animate-pulse">
                 <div className="bg-white border border-slate-100 px-5 py-4 rounded-2xl rounded-tl-none shadow-sm flex items-center gap-2">
                   <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                   <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                   <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                 </div>
               </div>
             )}
             <div ref={messagesEndRef} />
          </div>
        </div>
      </main>

      <InputArea 
        onSendText={handleSendText} 
        onSendVoice={handleSendVoice}
        isLoading={isLoading}
      />
    </div>
  );
};

export default App;
