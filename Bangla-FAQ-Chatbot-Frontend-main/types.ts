export interface TextRequest {
  question: string;
}

export interface TTSRequest {
  text: string;
}

export interface TTSResponse {
  audio: string; // Hex string
}

export interface BotResponse {
  category?: string;
  answer: string;
  question?: string; // For voice response echoing
}

export enum MessageSender {
  User = 'user',
  Bot = 'bot',
}

export interface Message {
  id: string;
  text: string;
  sender: MessageSender;
  category?: string;
  timestamp: Date;
  isAudioMessage?: boolean;
}

export enum RecorderStatus {
  Idle = 'idle',
  Recording = 'recording',
  Processing = 'processing',
}