import React, { useState, useRef } from 'react';
import { Volume2, VolumeX, Zap, Radio } from 'lucide-react';

export default function AudioTelemetryEngine() {
  const [isPlaying, setIsPlaying] = useState(false);
  const audioCtxRef = useRef(null);

  const startSound = () => {
    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      const ctx = new AudioContext();
      audioCtxRef.current = ctx;

      // Master Gain
      const masterGain = ctx.createGain();
      masterGain.gain.setValueAtTime(0.08, ctx.currentTime);
      masterGain.connect(ctx.destination);

      // 1. Engine V6 Fundamental Frequency Oscillator
      const osc1 = ctx.createOscillator();
      osc1.type = 'sawtooth';
      osc1.frequency.setValueAtTime(140, ctx.currentTime); // idle RPM

      // Frequency ramp simulating acceleration through gears
      osc1.frequency.exponentialRampToValueAtTime(380, ctx.currentTime + 1.2);
      osc1.frequency.setValueAtTime(220, ctx.currentTime + 1.25); // gear shift drop
      osc1.frequency.exponentialRampToValueAtTime(460, ctx.currentTime + 2.5);
      osc1.frequency.setValueAtTime(280, ctx.currentTime + 2.55); // gear shift drop
      osc1.frequency.exponentialRampToValueAtTime(520, ctx.currentTime + 3.8);

      // Lowpass filter for deep exhaust tone
      const filter = ctx.createBiquadFilter();
      filter.type = 'lowpass';
      filter.frequency.setValueAtTime(1200, ctx.currentTime);

      osc1.connect(filter);
      filter.connect(masterGain);

      // 2. High-rev turbo whine harmonic
      const turboOsc = ctx.createOscillator();
      turboOsc.type = 'sine';
      turboOsc.frequency.setValueAtTime(800, ctx.currentTime);
      turboOsc.frequency.exponentialRampToValueAtTime(2400, ctx.currentTime + 3.8);

      const turboGain = ctx.createGain();
      turboGain.gain.setValueAtTime(0.02, ctx.currentTime);
      turboOsc.connect(turboGain);
      turboGain.connect(masterGain);

      osc1.start();
      turboOsc.start();

      osc1.stop(ctx.currentTime + 4.2);
      turboOsc.stop(ctx.currentTime + 4.2);

      setIsPlaying(true);
      setTimeout(() => {
        setIsPlaying(false);
      }, 4200);
    } catch (e) {
      console.warn('Web Audio not supported or user gesture needed:', e);
    }
  };

  const stopSound = () => {
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    setIsPlaying(false);
  };

  return (
    <div className="flex items-center gap-2">
      <button
        onClick={isPlaying ? stopSound : startSound}
        className={`px-3 py-1.5 border text-xs font-label-caps uppercase flex items-center gap-2 transition-all ${
          isPlaying
            ? 'bg-primary-container text-text-primary border-primary-container shadow-[0_0_12px_rgba(225,6,0,0.5)]'
            : 'bg-surface-panel hover:bg-surface-raised border-border-subtle text-text-muted hover:text-text-primary'
        }`}
        title="Simulate 2027 Power Unit Acceleration Audio"
      >
        {isPlaying ? <Volume2 className="w-4 h-4 animate-bounce" /> : <VolumeX className="w-4 h-4 text-text-subtle" />}
        <span>{isPlaying ? 'ENGINE REV TELEMETRY ACTIVE (12,500 RPM)' : 'HEAR SF-27 PU REVS'}</span>
      </button>
    </div>
  );
}
