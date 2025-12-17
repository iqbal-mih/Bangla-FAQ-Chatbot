import { BotResponse, TTSResponse } from '../types';

// Using 127.0.0.1 instead of localhost avoids some common DNS resolution issues in certain environments
const BASE_URL = 'http://127.0.0.1:8000';

export const ApiService = {
  /**
   * Send text question to backend
   */
  askText: async (question: string): Promise<BotResponse> => {
    try {
      // Based on the generated cURL command, the backend expects the question as a query parameter
      const params = new URLSearchParams();
      params.append('question', question);

      const response = await fetch(`${BASE_URL}/ask_text?${params.toString()}`, {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
        },
        // No body required when sending data via query params
      });

      if (!response.ok) throw new Error('Failed to get answer');
      return await response.json();
    } catch (error) {
      console.error('API Error (askText):', error);
      throw error;
    }
  },

  /**
   * Send audio file to backend
   */
  askVoice: async (audioBlob: Blob): Promise<BotResponse> => {
    try {
      const formData = new FormData();
      // Backend expects 'file' parameter with a .wav extension
      const file = new File([audioBlob], 'recording.wav', { type: 'audio/wav' });
      formData.append('file', file);

      const response = await fetch(`${BASE_URL}/ask_voice`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Failed to process voice');
      return await response.json();
    } catch (error) {
      console.error('API Error (askVoice):', error);
      throw error;
    }
  },

  /**
   * Request Text-to-Speech audio
   */
  getTTS: async (text: string): Promise<string> => {
    try {
      const response = await fetch(`${BASE_URL}/tts`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) throw new Error('Failed to get TTS');
      const data: TTSResponse = await response.json();
      return data.audio; // Returns hex string
    } catch (error) {
      console.error('API Error (tts):', error);
      throw error;
    }
  },
};