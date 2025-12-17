// utils/wavEncoder.ts
// Utility to convert WebM/Opus/PCM audio Blob to WAV Blob

/**
 * Convert a WebM/Opus/PCM audio Blob to a WAV Blob
 * @param audioBlob The input audio Blob (from MediaRecorder)
 * @returns Promise<Blob> WAV Blob
 */
export async function convertToWav(audioBlob: Blob): Promise<Blob> {
  // Read audio data as ArrayBuffer
  const arrayBuffer = await audioBlob.arrayBuffer();
  // Decode audio data to PCM using AudioContext
  const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

  // Prepare WAV file parameters
  const numChannels = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const format = 1; // PCM
  const bitDepth = 16;
  const numSamples = audioBuffer.length;
  const blockAlign = numChannels * bitDepth / 8;
  const byteRate = sampleRate * blockAlign;
  const wavBuffer = new ArrayBuffer(44 + numSamples * blockAlign);
  const view = new DataView(wavBuffer);

  // Write WAV header
  let offset = 0;
  function writeString(s: string) {
    for (let i = 0; i < s.length; i++) view.setUint8(offset++, s.charCodeAt(i));
  }
  writeString('RIFF');
  view.setUint32(offset, 36 + numSamples * blockAlign, true); offset += 4;
  writeString('WAVE');
  writeString('fmt ');
  view.setUint32(offset, 16, true); offset += 4; // Subchunk1Size
  view.setUint16(offset, format, true); offset += 2; // AudioFormat
  view.setUint16(offset, numChannels, true); offset += 2;
  view.setUint32(offset, sampleRate, true); offset += 4;
  view.setUint32(offset, byteRate, true); offset += 4;
  view.setUint16(offset, blockAlign, true); offset += 2;
  view.setUint16(offset, bitDepth, true); offset += 2;
  writeString('data');
  view.setUint32(offset, numSamples * blockAlign, true); offset += 4;

  // Write PCM samples
  for (let ch = 0; ch < numChannels; ch++) {
    const channel = audioBuffer.getChannelData(ch);
    let sampleOffset = 44 + ch * 2;
    for (let i = 0; i < numSamples; i++, sampleOffset += blockAlign) {
      let sample = Math.max(-1, Math.min(1, channel[i]));
      sample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
      view.setInt16(44 + i * blockAlign + ch * 2, sample, true);
    }
  }

  return new Blob([wavBuffer], { type: 'audio/wav' });
}
