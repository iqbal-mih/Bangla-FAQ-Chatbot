/**
 * Converts a hex string to an Audio Blob url
 * @param hexString The hex string containing audio data
 * @returns Blob URL string
 */
export const hexToAudioUrl = (hexString: string): string | null => {
  if (!hexString) return null;
  
  try {
    const match = hexString.match(/[\da-f]{2}/gi);
    if (!match) return null;
    
    const bytes = new Uint8Array(match.map((h) => parseInt(h, 16)));
    const blob = new Blob([bytes], { type: 'audio/mp3' }); // Assuming backend sends MP3/WAV compatible binary
    return URL.createObjectURL(blob);
  } catch (e) {
    console.error("Error converting hex to audio:", e);
    return null;
  }
};

/**
 * Play audio from a URL
 */
export const playAudio = (url: string): Promise<void> => {
  return new Promise((resolve, reject) => {
    const audio = new Audio(url);
    audio.onended = () => resolve();
    audio.onerror = (e) => reject(e);
    audio.play().catch(reject);
  });
};
