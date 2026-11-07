import React, { useState } from 'react';
import { X, Play, ShieldAlert, Radio, Activity, Volume2, VolumeX } from 'lucide-react';

export default function VideoModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  const [activeClip, setActiveClip] = useState('onboard');

  const videoClips = [
    {
      id: 'onboard',
      title: 'Scuderia Ferrari SF-27 Onboard Telemetry (Monza High-Speed Lap)',
      desc: '354 km/h straight-line deployment, active aero DRS angle transition, and 5.2G braking telemetry through Variante del Rettifilo.',
      duration: '01:42',
      type: 'TELEMETRY STREAM',
      youtubeId: 'pS-W4fU7K80', // F1 onboard / Ferrari track action
    },
    {
      id: 'aero',
      title: 'Maranello Wind Tunnel & Active Aerodynamics Validation',
      desc: 'Computational Fluid Dynamics (CFD) correlation, front-wing ground effect ground clearance, and vortex wake dissipation.',
      duration: '02:15',
      type: 'AERODYNAMIC R&D',
      youtubeId: 'q7i9U97vQ9o',
    },
    {
      id: 'engine',
      title: '2027 50/50 Power Unit Dyno Test & 350kW MGU-K Boost',
      desc: '1.6L turbocharged V6 hybrid internal combustion engine coupled with 350kW electric motor deployment curve.',
      duration: '00:58',
      type: 'POWERTRAIN DYNO',
      youtubeId: 'W8p4b2G9P2I',
    },
  ];

  const current = videoClips.find((c) => c.id === activeClip) || videoClips[0];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-canvas-base/85 backdrop-blur-md">
      <div className="w-full max-w-4xl bg-surface-panel border border-border-focus shadow-2xl flex flex-col overflow-hidden">
        {/* Modal Header */}
        <div className="bg-canvas-base px-5 py-3 border-b border-border-subtle flex justify-between items-center text-xs font-label-caps">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-primary-container animate-pulse status-dot"></span>
            <span className="text-text-primary font-bold uppercase tracking-wider">
              SCUDERIA COMMAND // VIDEO DEMONSTRATION &amp; TELEMETRY STREAM
            </span>
          </div>
          <button
            onClick={onClose}
            className="p-1 text-text-muted hover:text-text-primary hover:bg-surface-raised transition-colors"
            aria-label="Close video player"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Video Player Display */}
        <div className="relative aspect-video w-full bg-black overflow-hidden border-b border-border-subtle flex items-center justify-center">
          {/* Active Video Embed */}
          <iframe
            className="w-full h-full"
            src={`https://www.youtube-nocookie.com/embed/${current.youtubeId}?autoplay=1&mute=0&controls=1&rel=0`}
            title={current.title}
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
          />

          {/* Telemetry HUD scanlines overlay effect */}
          <div className="absolute top-3 left-3 pointer-events-none text-[10px] font-label-caps uppercase bg-canvas-base/80 px-2 py-1 border border-border-subtle text-primary-container flex items-center gap-1.5">
            <Radio className="w-3 h-3 text-secondary animate-pulse" />
            <span>LIVE BROADCAST FEED: 1080p 60FPS</span>
          </div>
        </div>

        {/* Video Selector Strip */}
        <div className="p-4 bg-surface-raised flex flex-col gap-3">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
            <div>
              <span className="text-[10px] font-label-caps uppercase text-secondary font-bold">
                {current.type} · {current.duration}
              </span>
              <h4 className="font-headline-sm text-base uppercase text-text-primary font-bold">
                {current.title}
              </h4>
              <p className="text-xs text-text-muted mt-0.5">{current.desc}</p>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 pt-2 border-t border-border-subtle">
            {videoClips.map((clip) => (
              <button
                key={clip.id}
                onClick={() => setActiveClip(clip.id)}
                className={`p-2.5 text-left text-xs transition-colors flex items-center gap-2 border ${
                  activeClip === clip.id
                    ? 'bg-primary-container/20 border-primary-container text-text-primary font-bold'
                    : 'bg-surface-panel border-border-subtle text-text-muted hover:text-text-primary hover:bg-surface-bright'
                }`}
              >
                <Play className="w-3.5 h-3.5 text-primary-container shrink-0" />
                <div className="flex flex-col truncate">
                  <span className="truncate uppercase font-label-caps text-[10px]">{clip.title}</span>
                  <span className="text-[9px] text-text-subtle">{clip.type} ({clip.duration})</span>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
