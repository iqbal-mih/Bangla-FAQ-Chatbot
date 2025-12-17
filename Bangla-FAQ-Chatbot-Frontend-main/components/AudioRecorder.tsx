import React, { useRef, useState, useEffect } from 'react';
import { Mic, Square, Loader2 } from 'lucide-react';
import { convertToWav } from '../utils/wavEncoder';

interface AudioRecorderProps {
  onRecordingComplete: (blob: Blob) => void;
  isProcessing: boolean;
}

const AudioRecorder: React.FC<AudioRecorderProps> = ({ onRecordingComplete, isProcessing }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current);
    };
  }, []);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        // Convert to WAV before sending
        try {
          const wavBlob = await convertToWav(blob);
          onRecordingComplete(wavBlob);
        } catch (err) {
          console.error('WAV conversion failed, sending original blob:', err);
          onRecordingComplete(blob);
        }
        // Stop all tracks
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      
      // Timer
      setRecordingTime(0);
      timerRef.current = window.setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

    } catch (err) {
      console.error("Error accessing microphone:", err);
      alert("মাইক্রোফোন অ্যাক্সেস করতে সমস্যা হচ্ছে। অনুগ্রহ করে অনুমতি দিন।");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerRef.current) {
        window.clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="relative">
      {isRecording ? (
        <div className="flex items-center gap-3 bg-red-50 text-red-600 px-4 py-2 rounded-full border border-red-100 animate-pulse">
            <div className="w-2 h-2 bg-red-500 rounded-full animate-ping" />
            <span className="text-sm font-mono font-medium">{formatTime(recordingTime)}</span>
            <button 
                onClick={stopRecording}
                className="p-1.5 bg-red-100 hover:bg-red-200 rounded-full transition-colors"
                title="Stop Recording"
            >
                <Square size={16} fill="currentColor" />
            </button>
        </div>
      ) : (
        <button
          onClick={startRecording}
          disabled={isProcessing}
          className={`p-3 rounded-full transition-all duration-200 ${
            isProcessing 
              ? 'bg-slate-100 text-slate-400 cursor-not-allowed' 
              : 'bg-slate-100 text-slate-600 hover:bg-emerald-100 hover:text-emerald-600'
          }`}
          title="Speak Question"
        >
           {isProcessing ? <Loader2 size={20} className="animate-spin" /> : <Mic size={20} />}
        </button>
      )}
    </div>
  );
};

export default AudioRecorder;
