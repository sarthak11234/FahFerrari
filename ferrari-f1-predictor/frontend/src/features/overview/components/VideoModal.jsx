import React, { useState, useEffect, useRef } from 'react';
import {
  X,
  Play,
  Pause,
  RotateCcw,
  Volume2,
  VolumeX,
  Radio,
  ExternalLink,
  Activity,
  Gauge,
  Zap,
  Wind,
  Layers,
  ChevronRight,
  AlertTriangle
} from 'lucide-react';

export default function VideoModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  const [activeClip, setActiveClip] = useState('onboard');
  const [isPlaying, setIsPlaying] = useState(true);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [progress, setProgress] = useState(25); // 0 to 100%
  const [isAudioEnabled, setIsAudioEnabled] = useState(false);
  const [displayMode, setDisplayMode] = useState('interactive'); // 'interactive' | 'broadcast'

  const canvasRef = useRef(null);
  const animFrameRef = useRef(null);
  const audioCtxRef = useRef(null);
  const oscRef = useRef(null);
  const gainRef = useRef(null);

  const videoClips = [
    {
      id: 'onboard',
      title: 'Scuderia Ferrari SF-27 Onboard Telemetry (Monza High-Speed Lap)',
      desc: '354 km/h straight-line deployment, active aero DRS angle transition, and 5.2G braking telemetry through Variante del Rettifilo.',
      duration: '01:24',
      type: 'TELEMETRY SIMULATION',
      mode: 'interactive',
      track: 'Autodromo Nazionale Monza',
      lapTime: '1:19.428',
      youtubeId: 'eUHDENy1jH4', // Scuderia Ferrari Official Fire Up & Shakedown
      externalUrl: 'https://www.youtube.com/@Ferrari',
    },
    {
      id: 'aero',
      title: 'Maranello Wind Tunnel & Active Aerodynamics Validation',
      desc: 'Computational Fluid Dynamics (CFD) correlation, front-wing ground effect ground clearance, and active DRS vortex wake dissipation.',
      duration: '02:15',
      type: 'AERODYNAMIC R&D',
      mode: 'interactive',
      track: 'Galleria del Vento Maranello',
      lapTime: 'STEADY STATE',
      youtubeId: 'eUHDENy1jH4',
      externalUrl: 'https://www.youtube.com/@Ferrari',
    },
    {
      id: 'engine',
      title: '2027 50/50 Power Unit Dyno Test & 350kW MGU-K Boost',
      desc: '1.6L turbocharged V6 hybrid internal combustion engine coupled with 350kW electric motor deployment curve.',
      duration: '00:58',
      type: 'POWERTRAIN DYNO',
      mode: 'interactive',
      track: 'Scuderia Ferrari Dyno Cell 4',
      lapTime: 'RPM SWEEP',
      youtubeId: 'eUHDENy1jH4',
      externalUrl: 'https://www.youtube.com/@Ferrari',
    },
    {
      id: 'factory',
      title: 'Scuderia Ferrari Official Factory Broadcast (Maranello)',
      desc: 'Official Scuderia Ferrari presentation stream, garage telemetry operations, and SF-series technical shakedown.',
      duration: '03:40',
      type: 'OFFICIAL BROADCAST',
      mode: 'broadcast',
      track: 'Pista di Fiorano',
      lapTime: 'OFFICIAL FEED',
      youtubeId: 'eUHDENy1jH4',
      externalUrl: 'https://www.youtube.com/watch?v=eUHDENy1jH4',
    },
  ];

  const current = videoClips.find((c) => c.id === activeClip) || videoClips[0];

  // Calculate telemetry values along the Monza lap based on progress (0 - 100)
  const getTelemetryData = (pct) => {
    // 0-30: Prima Variante straight & chicane (accel to 354, brake to 76)
    // 30-55: Curva Grande & Roggia chicane (290 -> 110)
    // 55-75: Lesmo 1 & Lesmo 2 (165 -> 145)
    // 75-100: Serraglio & Parabolica onto main straight (335 -> 190 -> 320)
    let speed = 320;
    let gear = 7;
    let rpm = 11800;
    let throttle = 100;
    let brake = 0;
    let drs = true;
    let gLat = 0.2;
    let gLong = 0.8;
    let sector = 'SECTOR 1';

    if (pct < 20) {
      // Main straight acceleration
      speed = Math.floor(280 + (pct / 20) * 74); // 280 -> 354
      gear = 8;
      rpm = Math.floor(10500 + (pct / 20) * 1900);
      throttle = 100;
      brake = 0;
      drs = true;
      gLat = 0.1;
      gLong = 1.2;
      sector = 'SECTOR 1 (RETTIFILO)';
    } else if (pct < 28) {
      // Heavy braking for Variante del Rettifilo
      const brakePhase = (pct - 20) / 8;
      speed = Math.floor(354 - brakePhase * 278); // 354 -> 76
      gear = Math.max(2, 8 - Math.floor(brakePhase * 6));
      rpm = Math.floor(12400 - brakePhase * 4000);
      throttle = 0;
      brake = Math.floor(100 - brakePhase * 30);
      drs = false;
      gLat = 0.4;
      gLong = -5.2; // 5.2G braking
      sector = 'SECTOR 1 (CHICANE APEX)';
    } else if (pct < 45) {
      // Accelerating through Curva Grande
      const cgPhase = (pct - 28) / 17;
      speed = Math.floor(76 + cgPhase * 224); // 76 -> 300
      gear = Math.min(7, 2 + Math.floor(cgPhase * 5));
      rpm = Math.floor(9000 + cgPhase * 3000);
      throttle = Math.floor(40 + cgPhase * 60);
      brake = 0;
      drs = cgPhase > 0.6;
      gLat = 3.8; // lateral load in Curva Grande
      gLong = 1.4;
      sector = 'SECTOR 1 (CURVA GRANDE)';
    } else if (pct < 65) {
      // Variante della Roggia & Lesmo
      const lPhase = (pct - 45) / 20;
      speed = Math.floor(220 - Math.sin(lPhase * Math.PI) * 80);
      gear = 4;
      rpm = 10800;
      throttle = 65;
      brake = lPhase < 0.3 ? 60 : 0;
      drs = false;
      gLat = 4.2;
      gLong = -1.8;
      sector = 'SECTOR 2 (LESMO 1 & 2)';
    } else {
      // Serraglio & Parabolica / Curva Alboreto
      const pPhase = (pct - 65) / 35;
      speed = Math.floor(240 + pPhase * 95);
      gear = 7;
      rpm = Math.floor(11000 + pPhase * 1400);
      throttle = Math.floor(70 + pPhase * 30);
      brake = 0;
      drs = pPhase > 0.7;
      gLat = 2.9;
      gLong = 1.1;
      sector = 'SECTOR 3 (CURVA ALBORETO)';
    }

    return { speed, gear, rpm, throttle, brake, drs, gLat, gLong, sector };
  };

  const telemetry = getTelemetryData(progress);

  // Playback timer loop
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setProgress((prev) => {
        const next = prev + 0.3 * playbackSpeed;
        return next > 100 ? 0 : next;
      });
    }, 50);

    return () => clearInterval(interval);
  }, [isPlaying, playbackSpeed]);

  // Web Audio synthesizer for engine sounds
  useEffect(() => {
    if (!isAudioEnabled) {
      if (audioCtxRef.current) {
        audioCtxRef.current.close().catch(() => {});
        audioCtxRef.current = null;
        oscRef.current = null;
      }
      return;
    }

    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      const ctx = new AudioContext();
      audioCtxRef.current = ctx;

      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      const filter = ctx.createBiquadFilter();

      osc.type = 'sawtooth';
      filter.type = 'lowpass';
      filter.frequency.setValueAtTime(1400, ctx.currentTime);

      gain.gain.setValueAtTime(0.04, ctx.currentTime);

      osc.connect(filter);
      filter.connect(gain);
      gain.connect(ctx.destination);

      osc.start();
      oscRef.current = osc;
      gainRef.current = gain;
    } catch (e) {
      console.warn('AudioContext error:', e);
    }

    return () => {
      if (audioCtxRef.current) {
        audioCtxRef.current.close().catch(() => {});
        audioCtxRef.current = null;
      }
    };
  }, [isAudioEnabled]);

  // Adjust pitch with telemetry RPM
  useEffect(() => {
    if (oscRef.current && audioCtxRef.current && isAudioEnabled) {
      const freq = 120 + (telemetry.rpm / 12500) * 380;
      oscRef.current.frequency.setTargetAtTime(freq, audioCtxRef.current.currentTime, 0.05);
    }
  }, [telemetry.rpm, isAudioEnabled]);

  // Canvas 60FPS Video Telemetry Renderer
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let frame = 0;

    const render = () => {
      frame++;
      const w = canvas.width;
      const h = canvas.height;

      // Background Asphalt & Sky
      ctx.fillStyle = '#0a0b0e';
      ctx.fillRect(0, 0, w, h);

      // Horizon line
      const horizonY = h * 0.42;

      // Sky gradient / night atmosphere
      const skyGrad = ctx.createLinearGradient(0, 0, 0, horizonY);
      skyGrad.addColorStop(0, '#040507');
      skyGrad.addColorStop(1, '#11151c');
      ctx.fillStyle = skyGrad;
      ctx.fillRect(0, 0, w, horizonY);

      // Distant track lights & grandstands
      ctx.fillStyle = '#1c222b';
      ctx.fillRect(0, horizonY - 14, w, 14);
      for (let i = 0; i < w; i += 40) {
        ctx.fillStyle = (i + Math.floor(frame * 0.2)) % 80 === 0 ? '#55C58A' : '#e7b45a';
        ctx.fillRect(i, horizonY - 12, 3, 2);
      }

      // Track surface
      const trackGrad = ctx.createLinearGradient(0, horizonY, 0, h);
      trackGrad.addColorStop(0, '#13161c');
      trackGrad.addColorStop(1, '#08090b');
      ctx.fillStyle = trackGrad;
      ctx.fillRect(0, horizonY, w, h - horizonY);

      // 3D Perspective Road & Kerbs
      const roadSpeed = (telemetry.speed / 100) * playbackSpeed;
      const offset = (frame * roadSpeed * 3) % 60;

      ctx.save();
      const centerX = w / 2;
      const topWidth = 80;
      const botWidth = w * 0.85;

      // Road outline
      ctx.beginPath();
      ctx.moveTo(centerX - topWidth / 2, horizonY);
      ctx.lineTo(centerX + topWidth / 2, horizonY);
      ctx.lineTo(centerX + botWidth / 2, h);
      ctx.lineTo(centerX - botWidth / 2, h);
      ctx.closePath();
      ctx.fillStyle = '#171a21';
      ctx.fill();

      // Red & White Kerbs
      const segments = 16;
      for (let i = 0; i < segments; i++) {
        const t1 = i / segments;
        const t2 = (i + 1) / segments;
        const y1 = horizonY + (h - horizonY) * Math.pow(t1, 1.8);
        const y2 = horizonY + (h - horizonY) * Math.pow(t2, 1.8);

        const w1 = topWidth + (botWidth - topWidth) * t1;
        const w2 = topWidth + (botWidth - topWidth) * t2;

        const isRed = (i + Math.floor(offset / 15)) % 2 === 0;
        ctx.fillStyle = isRed ? '#E10600' : '#f4f4f2';

        // Left kerb
        ctx.beginPath();
        ctx.moveTo(centerX - w1 / 2, y1);
        ctx.lineTo(centerX - w1 / 2 - 20 * t1, y1);
        ctx.lineTo(centerX - w2 / 2 - 20 * t2, y2);
        ctx.lineTo(centerX - w2 / 2, y2);
        ctx.closePath();
        ctx.fill();

        // Right kerb
        ctx.beginPath();
        ctx.moveTo(centerX + w1 / 2, y1);
        ctx.lineTo(centerX + w1 / 2 + 20 * t1, y1);
        ctx.lineTo(centerX + w2 / 2 + 20 * t2, y2);
        ctx.lineTo(centerX + w2 / 2, y2);
        ctx.closePath();
        ctx.fill();
      }

      // Center dash lines
      for (let i = 0; i < segments; i += 2) {
        const t1 = i / segments;
        const t2 = (i + 1) / segments;
        const y1 = horizonY + (h - horizonY) * Math.pow(t1, 1.8);
        const y2 = horizonY + (h - horizonY) * Math.pow(t2, 1.8);
        ctx.strokeStyle = '#f4f4f2';
        ctx.lineWidth = Math.max(1, 4 * t2);
        ctx.beginPath();
        ctx.moveTo(centerX, y1);
        ctx.lineTo(centerX, y2);
        ctx.stroke();
      }

      // Speed blur streaks when speed > 280
      if (telemetry.speed > 280) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.lineWidth = 1.5;
        for (let j = 0; j < 8; j++) {
          const rx = (j * 117 + frame * 9) % w;
          const ry = horizonY + ((j * 73 + frame * 15) % (h - horizonY));
          ctx.beginPath();
          ctx.moveTo(rx, ry);
          ctx.lineTo(rx - (rx - centerX) * 0.15, ry - 18);
          ctx.stroke();
        }
      }

      // Ferrari SF-27 Cockpit Nosecone & Halo Frame
      const carBaseY = h - 20;
      const noseTopY = h - 120;

      // Nosecone
      ctx.fillStyle = '#E10600';
      ctx.beginPath();
      ctx.moveTo(centerX - 42, carBaseY);
      ctx.lineTo(centerX - 18, noseTopY);
      ctx.lineTo(centerX + 18, noseTopY);
      ctx.lineTo(centerX + 42, carBaseY);
      ctx.closePath();
      ctx.fill();

      // Yellow Modena stripe on nose
      ctx.fillStyle = '#FFEB3B';
      ctx.fillRect(centerX - 2, noseTopY, 4, carBaseY - noseTopY);

      // Scuderia Ferrari Carbon Antenna / Pitot Tube
      ctx.strokeStyle = '#292D33';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(centerX, noseTopY);
      ctx.lineTo(centerX, noseTopY - 24);
      ctx.stroke();

      // Halo titanium center pylon
      ctx.strokeStyle = '#111317';
      ctx.lineWidth = 8;
      ctx.beginPath();
      ctx.moveTo(centerX, h);
      ctx.lineTo(centerX, h - 80);
      ctx.stroke();

      // Active Aero Flow Streamers in Aero mode
      if (activeClip === 'aero') {
        ctx.strokeStyle = 'rgba(85, 197, 138, 0.65)';
        ctx.lineWidth = 1.5;
        for (let a = 0; a < 10; a++) {
          const sy = horizonY + a * 18;
          ctx.beginPath();
          ctx.moveTo(0, sy);
          ctx.bezierCurveTo(
            w * 0.3,
            sy + Math.sin(frame * 0.1 + a) * 12,
            w * 0.7,
            sy - Math.cos(frame * 0.1 + a) * 12,
            w,
            sy
          );
          ctx.stroke();
        }
      }

      ctx.restore();

      animFrameRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [telemetry.speed, playbackSpeed, activeClip]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-6 bg-canvas-base/90 backdrop-blur-md">
      <div className="w-full max-w-5xl bg-surface-panel border border-border-focus shadow-2xl flex flex-col overflow-hidden max-h-[92vh]">
        {/* Modal Top Header */}
        <div className="bg-canvas-base px-4 sm:px-6 py-3 border-b border-border-subtle flex justify-between items-center text-xs font-label-caps">
          <div className="flex items-center gap-3">
            <span className="w-2.5 h-2.5 rounded-full bg-primary-container animate-pulse status-dot shadow-[0_0_8px_#E10600]"></span>
            <span className="text-text-primary font-bold uppercase tracking-wider text-sm">
              SCUDERIA COMMAND // VIRTUAL TELEMETRY STREAM &amp; ONBOARD LAB
            </span>
            <span className="hidden md:inline-block px-2 py-0.5 bg-surface-raised border border-border-subtle text-[10px] text-secondary font-mono">
              60 FPS DIRECT SIMULATION
            </span>
          </div>

          <div className="flex items-center gap-3">
            {/* Audio Toggle */}
            <button
              onClick={() => setIsAudioEnabled(!isAudioEnabled)}
              className={`px-2.5 py-1 text-xs font-mono uppercase flex items-center gap-1.5 border transition-all ${
                isAudioEnabled
                  ? 'bg-secondary/20 text-secondary border-secondary shadow-[0_0_8px_rgba(85,197,138,0.4)]'
                  : 'bg-surface-raised text-text-muted border-border-subtle hover:text-text-primary'
              }`}
              title="Toggle Simulated Engine Acceleration Sound"
            >
              {isAudioEnabled ? <Volume2 className="w-3.5 h-3.5 animate-pulse" /> : <VolumeX className="w-3.5 h-3.5" />}
              <span className="hidden sm:inline">{isAudioEnabled ? 'AUDIO ON' : 'AUDIO OFF'}</span>
            </button>

            {/* Close Button */}
            <button
              onClick={onClose}
              className="p-1 text-text-muted hover:text-text-primary hover:bg-surface-raised transition-colors"
              aria-label="Close telemetry player"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Video Player Display Canvas & Overlays */}
        <div className="relative aspect-video w-full bg-black overflow-hidden border-b border-border-subtle flex items-center justify-center select-none">
          {displayMode === 'interactive' ? (
            <>
              {/* 60 FPS HTML5 Canvas Simulation Renderer */}
              <canvas
                ref={canvasRef}
                width={960}
                height={540}
                className="w-full h-full object-cover"
              />

              {/* F1 HUD TELEMETRY OVERLAY */}
              <div className="absolute inset-0 pointer-events-none p-4 flex flex-col justify-between">
                {/* Top Telemetry Header */}
                <div className="flex justify-between items-start">
                  <div className="flex flex-col gap-1">
                    <div className="flex items-center gap-2 bg-canvas-base/85 backdrop-blur-sm px-2.5 py-1 border border-border-subtle">
                      <Radio className="w-3.5 h-3.5 text-primary-container animate-pulse" />
                      <span className="text-[11px] font-label-caps text-text-primary font-bold">
                        ONBOARD CAM 01 // CHARLES LECLERC [16]
                      </span>
                    </div>
                    <div className="text-[10px] font-mono text-secondary px-2.5 py-0.5 bg-canvas-base/75 border border-border-subtle w-fit">
                      {telemetry.sector}
                    </div>
                  </div>

                  {/* DRS & Lap Status */}
                  <div className="flex flex-col items-end gap-1">
                    <div
                      className={`px-3 py-1 text-xs font-label-caps font-bold uppercase tracking-wider border ${
                        telemetry.drs
                          ? 'bg-secondary text-canvas-base border-secondary shadow-[0_0_12px_rgba(85,197,138,0.7)] animate-pulse'
                          : 'bg-surface-panel/85 text-text-muted border-border-subtle'
                      }`}
                    >
                      {telemetry.drs ? 'DRS ENABLED' : 'DRS CLOSED'}
                    </div>
                    <div className="text-[10px] font-mono text-text-muted bg-canvas-base/80 px-2 py-0.5 border border-border-subtle">
                      REF LAP: {current.lapTime}
                    </div>
                  </div>
                </div>

                {/* Bottom HUD: Speed, Gear, RPM, Throttle, Brake */}
                <div className="flex justify-between items-end">
                  {/* Left: Speedometer & Gear */}
                  <div className="flex items-end gap-3 bg-canvas-base/85 backdrop-blur-md p-3 border border-border-subtle">
                    {/* Big Gear */}
                    <div className="flex flex-col items-center">
                      <span className="text-[9px] font-label-caps text-text-subtle">GEAR</span>
                      <span className="font-headline-sm text-4xl text-text-primary font-bold leading-none font-mono">
                        {telemetry.gear}
                      </span>
                    </div>

                    {/* Speed km/h */}
                    <div className="flex flex-col border-l border-border-subtle pl-3">
                      <div className="flex items-baseline gap-1">
                        <span className="font-headline-sm text-4xl text-primary-container font-bold leading-none font-mono">
                          {telemetry.speed}
                        </span>
                        <span className="text-xs font-label-caps text-text-muted">KM/H</span>
                      </div>
                      <span className="text-[10px] font-mono text-text-subtle mt-0.5">
                        RPM: {telemetry.rpm.toLocaleString()} / 12,500
                      </span>
                    </div>
                  </div>

                  {/* Center: Shift Lights Bar */}
                  <div className="hidden sm:flex flex-col items-center gap-1 bg-canvas-base/80 px-3 py-2 border border-border-subtle">
                    <span className="text-[8px] font-mono text-text-subtle tracking-widest">REV LIMITER LEDS</span>
                    <div className="flex gap-1">
                      {[...Array(12)].map((_, idx) => {
                        const active = telemetry.rpm > 9000 + idx * 280;
                        let color = 'bg-secondary';
                        if (idx >= 6) color = 'bg-amber-400';
                        if (idx >= 9) color = 'bg-primary-container';
                        return (
                          <div
                            key={idx}
                            className={`w-2.5 h-3 border border-border-subtle ${
                              active ? color : 'bg-surface-panel/40'
                            }`}
                          />
                        );
                      })}
                    </div>
                  </div>

                  {/* Right: Throttle & Brake Bars + G-Force */}
                  <div className="flex items-center gap-3 bg-canvas-base/85 backdrop-blur-md p-3 border border-border-subtle">
                    {/* Throttle Bar */}
                    <div className="flex flex-col items-center">
                      <span className="text-[9px] font-label-caps text-secondary font-bold">THR</span>
                      <div className="w-3 h-16 bg-surface-raised border border-border-subtle relative flex items-end">
                        <div
                          className="w-full bg-secondary transition-all"
                          style={{ height: `${telemetry.throttle}%` }}
                        />
                      </div>
                      <span className="text-[9px] font-mono text-text-muted mt-1">{telemetry.throttle}%</span>
                    </div>

                    {/* Brake Bar */}
                    <div className="flex flex-col items-center">
                      <span className="text-[9px] font-label-caps text-primary-container font-bold">BRK</span>
                      <div className="w-3 h-16 bg-surface-raised border border-border-subtle relative flex items-end">
                        <div
                          className="w-full bg-primary-container transition-all"
                          style={{ height: `${telemetry.brake}%` }}
                        />
                      </div>
                      <span className="text-[9px] font-mono text-text-muted mt-1">{telemetry.brake}%</span>
                    </div>

                    {/* G-Force Box */}
                    <div className="hidden md:flex flex-col items-center pl-2 border-l border-border-subtle">
                      <span className="text-[9px] font-label-caps text-text-subtle">G-FORCE</span>
                      <div className="w-12 h-12 border border-border-subtle relative flex items-center justify-center bg-surface-raised/40">
                        {/* Crosshairs */}
                        <div className="absolute w-full h-px bg-border-subtle" />
                        <div className="absolute h-full w-px bg-border-subtle" />
                        {/* G-Ball */}
                        <div
                          className="w-2.5 h-2.5 rounded-full bg-primary-container shadow-[0_0_6px_#E10600] transition-all"
                          style={{
                            transform: `translate(${Math.min(18, Math.max(-18, telemetry.gLat * 4))}px, ${Math.min(
                              18,
                              Math.max(-18, -telemetry.gLong * 3)
                            )}px)`,
                          }}
                        />
                      </div>
                      <span className="text-[9px] font-mono text-text-primary mt-1">
                        {telemetry.gLong > 0 ? `+${telemetry.gLong}G` : `${telemetry.gLong}G`}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </>
          ) : (
            /* Broadcast Stream Embed Mode */
            <div className="relative w-full h-full flex flex-col items-center justify-center bg-canvas-base">
              <iframe
                className="w-full h-full"
                src={`https://www.youtube-nocookie.com/embed/${current.youtubeId}?autoplay=1&mute=1&controls=1&rel=0`}
                title={current.title}
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              />
              <div className="absolute bottom-3 right-3 bg-canvas-base/90 p-2 border border-border-subtle flex items-center gap-2">
                <a
                  href={current.externalUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-3 py-1 bg-primary-container text-text-primary text-xs font-label-caps flex items-center gap-1.5 hover:bg-primary-hover transition-colors"
                >
                  <span>WATCH ON YOUTUBE (OFFICIAL)</span>
                  <ExternalLink className="w-3.5 h-3.5" />
                </a>
              </div>
            </div>
          )}
        </div>

        {/* Playback Controls & Progress Scrubber */}
        <div className="bg-canvas-base px-4 py-3 border-b border-border-subtle flex flex-col gap-2">
          {/* Progress Scrubber */}
          <div className="flex items-center gap-3">
            <span className="text-[10px] font-mono text-text-subtle w-10">
              {Math.floor((progress * 84) / 100 / 60)}:
              {String(Math.floor(((progress * 84) / 100) % 60)).padStart(2, '0')}
            </span>
            <input
              type="range"
              min="0"
              max="100"
              step="0.1"
              value={progress}
              onChange={(e) => setProgress(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-surface-raised accent-primary-container cursor-pointer"
            />
            <span className="text-[10px] font-mono text-text-subtle w-10">01:24</span>
          </div>

          {/* Controls Bar */}
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-2">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="px-3 py-1 bg-surface-raised border border-border-subtle text-text-primary text-xs font-label-caps flex items-center gap-1.5 hover:bg-surface-bright transition-colors"
              >
                {isPlaying ? <Pause className="w-3.5 h-3.5 text-primary-container" /> : <Play className="w-3.5 h-3.5 text-secondary" />}
                <span>{isPlaying ? 'PAUSE' : 'PLAY'}</span>
              </button>

              <button
                onClick={() => setProgress(0)}
                className="p-1 bg-surface-raised border border-border-subtle text-text-muted hover:text-text-primary"
                title="Rewind to Lap Start"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>

              {/* Speed Buttons */}
              <div className="flex border border-border-subtle text-[10px] font-mono">
                {[0.5, 1, 2].map((spd) => (
                  <button
                    key={spd}
                    onClick={() => setPlaybackSpeed(spd)}
                    className={`px-2 py-0.5 ${
                      playbackSpeed === spd
                        ? 'bg-primary-container text-text-primary font-bold'
                        : 'bg-surface-raised text-text-muted hover:text-text-primary'
                    }`}
                  >
                    {spd}x
                  </button>
                ))}
              </div>
            </div>

            {/* Mode Switcher */}
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-label-caps text-text-subtle uppercase">MODE:</span>
              <button
                onClick={() => setDisplayMode('interactive')}
                className={`px-2.5 py-1 text-xs font-label-caps uppercase border transition-colors ${
                  displayMode === 'interactive'
                    ? 'bg-primary-container text-text-primary border-primary-container font-bold'
                    : 'bg-surface-raised text-text-muted border-border-subtle hover:text-text-primary'
                }`}
              >
                60FPS SIMULATION
              </button>
              <button
                onClick={() => setDisplayMode('broadcast')}
                className={`px-2.5 py-1 text-xs font-label-caps uppercase border transition-colors ${
                  displayMode === 'broadcast'
                    ? 'bg-primary-container text-text-primary border-primary-container font-bold'
                    : 'bg-surface-raised text-text-muted border-border-subtle hover:text-text-primary'
                }`}
              >
                YOUTUBE BROADCAST
              </button>
            </div>
          </div>
        </div>

        {/* Video Selector Strip */}
        <div className="p-4 bg-surface-raised flex flex-col gap-3">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
            <div>
              <span className="text-[10px] font-label-caps uppercase text-secondary font-bold">
                {current.type} · {current.duration} · {current.track}
              </span>
              <h4 className="font-headline-sm text-base uppercase text-text-primary font-bold">
                {current.title}
              </h4>
              <p className="text-xs text-text-muted mt-0.5">{current.desc}</p>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-2 pt-2 border-t border-border-subtle">
            {videoClips.map((clip) => (
              <button
                key={clip.id}
                onClick={() => {
                  setActiveClip(clip.id);
                  setDisplayMode(clip.mode);
                  setProgress(0);
                  setIsPlaying(true);
                }}
                className={`p-2.5 text-left text-xs transition-colors flex items-center gap-2 border ${
                  activeClip === clip.id
                    ? 'bg-primary-container/20 border-primary-container text-text-primary font-bold shadow-[0_0_8px_rgba(225,6,0,0.3)]'
                    : 'bg-surface-panel border-border-subtle text-text-muted hover:text-text-primary hover:bg-surface-bright'
                }`}
              >
                <Play className="w-3.5 h-3.5 text-primary-container shrink-0" />
                <div className="flex flex-col truncate">
                  <span className="truncate uppercase font-label-caps text-[10px]">{clip.title}</span>
                  <span className="text-[9px] text-text-subtle font-mono">
                    {clip.type} ({clip.duration})
                  </span>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
