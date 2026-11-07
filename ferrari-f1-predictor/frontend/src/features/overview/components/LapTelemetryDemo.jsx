import React, { useState, useEffect } from 'react';
import { Activity, Play, Pause, RotateCcw, Zap, Compass, Flag, Sliders } from 'lucide-react';

export default function LapTelemetryDemo() {
  const [distanceMeters, setDistanceMeters] = useState(1200);
  const [isPlaying, setIsPlaying] = useState(false);

  const lapPoints = [
    { dist: 0, corner: 'Start/Finish Straight', sec: 1, lecSpeed: 348, hamSpeed: 346, lecThr: 100, hamThr: 100, lecBrk: 0, hamBrk: 0, lecGear: 8, hamGear: 8, delta: '0.000s' },
    { dist: 600, corner: 'T1 Braking Zone (Rettifilo)', sec: 1, lecSpeed: 354, hamSpeed: 352, lecThr: 0, hamThr: 0, lecBrk: 115, hamBrk: 108, lecGear: 8, hamGear: 8, delta: '+0.012s' },
    { dist: 900, corner: 'T1 Apex (Variante del Rettifilo)', sec: 1, lecSpeed: 74, hamSpeed: 78, lecThr: 20, hamThr: 35, lecBrk: 40, hamBrk: 10, lecGear: 2, hamGear: 2, delta: '-0.035s' },
    { dist: 1300, corner: 'T3 Curva Grande', sec: 1, lecSpeed: 312, hamSpeed: 308, lecThr: 100, hamThr: 100, lecBrk: 0, hamBrk: 0, lecGear: 7, hamGear: 7, delta: '+0.024s' },
    { dist: 2100, corner: 'T4 Braking (Variante della Roggia)', sec: 2, lecSpeed: 336, hamSpeed: 334, lecThr: 0, hamThr: 0, lecBrk: 105, hamBrk: 98, lecGear: 7, hamGear: 7, delta: '+0.041s' },
    { dist: 2400, corner: 'T5 Roggia Apex', sec: 2, lecSpeed: 112, hamSpeed: 116, lecThr: 45, hamThr: 55, lecBrk: 0, hamBrk: 0, lecGear: 3, hamGear: 3, delta: '-0.015s' },
    { dist: 3000, corner: 'T6 Curva di Lesmo 1', sec: 2, lecSpeed: 264, hamSpeed: 260, lecThr: 85, hamThr: 80, lecBrk: 25, hamBrk: 30, lecGear: 5, hamGear: 5, delta: '+0.052s' },
    { dist: 3500, corner: 'T7 Curva di Lesmo 2', sec: 2, lecSpeed: 220, hamSpeed: 224, lecThr: 90, hamThr: 95, lecBrk: 15, hamBrk: 10, lecGear: 4, hamGear: 4, delta: '+0.028s' },
    { dist: 4200, corner: 'T8 Variante Ascari Entry', sec: 3, lecSpeed: 238, hamSpeed: 232, lecThr: 65, hamThr: 60, lecBrk: 45, hamBrk: 50, lecGear: 5, hamGear: 5, delta: '+0.071s' },
    { dist: 4700, corner: 'T10 Ascari Exit', sec: 3, lecSpeed: 288, hamSpeed: 292, lecThr: 100, hamThr: 100, lecBrk: 0, hamBrk: 0, lecGear: 6, hamGear: 6, delta: '+0.038s' },
    { dist: 5200, corner: 'T11 Curva Parabolica', sec: 3, lecSpeed: 215, hamSpeed: 218, lecThr: 75, hamThr: 85, lecBrk: 20, hamBrk: 15, lecGear: 4, hamGear: 4, delta: '-0.018s' },
    { dist: 5793, corner: 'Finish Line (Parabolica Exit)', sec: 3, lecSpeed: 342, hamSpeed: 345, lecThr: 100, hamThr: 100, lecBrk: 0, hamBrk: 0, lecGear: 8, hamGear: 8, delta: '+0.037s' },
  ];

  // Continuous animation loop when playing
  useEffect(() => {
    let interval;
    if (isPlaying) {
      interval = setInterval(() => {
        setDistanceMeters((prev) => {
          if (prev >= 5793) {
            setIsPlaying(false);
            return 5793;
          }
          return prev + 80;
        });
      }, 50);
    }
    return () => clearInterval(interval);
  }, [isPlaying]);

  // Find nearest telemetry point
  const currentTelemetry = lapPoints.reduce((prev, curr) => {
    return Math.abs(curr.dist - distanceMeters) < Math.abs(prev.dist - distanceMeters) ? curr : prev;
  });

  return (
    <div className="bg-surface-panel p-6 border border-border-subtle flex flex-col gap-6">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-border-subtle pb-4">
        <div>
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-primary-container" />
            <span className="font-label-caps text-xs uppercase text-primary-container tracking-wider font-bold">
              LIVE LAP TELEMETRY DEMONSTRATION // AUTODROMO NAZIONALE MONZA
            </span>
          </div>
          <h3 className="font-headline-md text-2xl uppercase text-text-primary font-bold mt-1">
            CHARLES LECLERC VS LEWIS HAMILTON · QUALIFYING TELEMETRY TRACE
          </h3>
          <p className="font-body-sm text-xs text-text-muted mt-0.5">
            Synchronized 50Hz telemetry trace over 5,793 meters. Scrutinize throttle percentage, brake pressure, speed apexes, and differential lap delta.
          </p>
        </div>

        {/* Telemetry Transport Controls */}
        <div className="flex items-center gap-2 bg-surface-raised p-1.5 border border-border-subtle">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="px-3 py-1 bg-primary-container hover:bg-ferrari-red-hover text-text-primary font-label-caps text-xs uppercase font-bold flex items-center gap-1.5 transition-colors"
          >
            {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            <span>{isPlaying ? 'PAUSE' : 'PLAY LAP'}</span>
          </button>
          <button
            onClick={() => {
              setIsPlaying(false);
              setDistanceMeters(0);
            }}
            className="p-1.5 bg-surface-panel hover:bg-surface-bright text-text-muted hover:text-text-primary transition-colors"
            title="Reset to Start/Finish"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
          <span className="font-data-mono-md text-xs text-text-primary px-2 font-bold tnum">
            {distanceMeters}m / 5,793m
          </span>
        </div>
      </div>

      {/* Scrubber Timeline Track */}
      <div className="flex flex-col gap-1.5">
        <div className="flex justify-between items-center text-xs font-label-caps text-text-subtle">
          <span className="text-primary-container font-bold uppercase">{currentTelemetry.corner}</span>
          <span>SECTOR {currentTelemetry.sec} · DELTA: <strong className="text-secondary font-data-mono-md">{currentTelemetry.delta} (LEC)</strong></span>
        </div>
        <input
          type="range"
          min="0"
          max="5793"
          step="20"
          value={distanceMeters}
          onChange={(e) => {
            setIsPlaying(false);
            setDistanceMeters(parseInt(e.target.value, 10));
          }}
          className="w-full accent-primary-container cursor-pointer bg-surface-raised h-3 border border-border-subtle"
        />
        <div className="flex justify-between text-[10px] font-label-caps text-text-subtle uppercase">
          <span>0m (Straight)</span>
          <span>1,200m (Curva Grande)</span>
          <span>2,400m (Roggia)</span>
          <span>3,500m (Lesmo)</span>
          <span>4,700m (Ascari)</span>
          <span>5,793m (Parabolica)</span>
        </div>
      </div>

      {/* Side-by-side Driver Comparison Telemetry Gauges */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Charles Leclerc (#16) */}
        <div className="bg-surface-raised p-5 border border-border-subtle border-l-4 border-l-primary-container flex flex-col gap-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-2">
              <span className="font-headline-lg text-2xl font-bold text-primary-container">#16</span>
              <div className="flex flex-col">
                <span className="font-headline-sm uppercase text-text-primary font-bold">CHARLES LECLERC</span>
                <span className="text-[10px] font-label-caps text-text-subtle">SCUDERIA FERRARI · SF-27</span>
              </div>
            </div>
            <div className="text-right">
              <span className="text-[10px] font-label-caps text-text-subtle uppercase block">TELEMETRY SPEED</span>
              <span className="font-data-mono-lg text-2xl text-primary-container font-bold tnum">
                {currentTelemetry.lecSpeed} <span className="text-xs font-normal text-text-muted">km/h</span>
              </span>
            </div>
          </div>

          {/* Bar Gauges */}
          <div className="flex flex-col gap-2.5 text-xs">
            <div>
              <div className="flex justify-between text-[10px] font-label-caps text-text-subtle mb-1">
                <span>THROTTLE POSITION</span>
                <span className="font-data-mono-md text-text-primary font-bold">{currentTelemetry.lecThr}%</span>
              </div>
              <div className="w-full bg-canvas-base h-2 border border-border-subtle">
                <div className="h-full bg-secondary transition-all" style={{ width: `${currentTelemetry.lecThr}%` }} />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-[10px] font-label-caps text-text-subtle mb-1">
                <span>BRAKE PRESSURE</span>
                <span className="font-data-mono-md text-text-primary font-bold">{currentTelemetry.lecBrk} BAR</span>
              </div>
              <div className="w-full bg-canvas-base h-2 border border-border-subtle">
                <div className="h-full bg-primary-container transition-all" style={{ width: `${Math.min(100, (currentTelemetry.lecBrk / 120) * 100)}%` }} />
              </div>
            </div>

            <div className="flex justify-between items-center pt-2 border-t border-border-subtle text-xs">
              <span className="text-text-subtle font-label-caps text-[10px] uppercase">GEARBOX ENGAGED:</span>
              <span className="font-data-mono-md text-base text-text-primary font-bold">GEAR {currentTelemetry.lecGear}</span>
            </div>
          </div>
        </div>

        {/* Lewis Hamilton (#44) */}
        <div className="bg-surface-raised p-5 border border-border-subtle border-l-4 border-l-telemetry-amber flex flex-col gap-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-2">
              <span className="font-headline-lg text-2xl font-bold text-telemetry-amber">#44</span>
              <div className="flex flex-col">
                <span className="font-headline-sm uppercase text-text-primary font-bold">LEWIS HAMILTON</span>
                <span className="text-[10px] font-label-caps text-text-subtle">SCUDERIA FERRARI · SF-27</span>
              </div>
            </div>
            <div className="text-right">
              <span className="text-[10px] font-label-caps text-text-subtle uppercase block">TELEMETRY SPEED</span>
              <span className="font-data-mono-lg text-2xl text-telemetry-amber font-bold tnum">
                {currentTelemetry.hamSpeed} <span className="text-xs font-normal text-text-muted">km/h</span>
              </span>
            </div>
          </div>

          {/* Bar Gauges */}
          <div className="flex flex-col gap-2.5 text-xs">
            <div>
              <div className="flex justify-between text-[10px] font-label-caps text-text-subtle mb-1">
                <span>THROTTLE POSITION</span>
                <span className="font-data-mono-md text-text-primary font-bold">{currentTelemetry.hamThr}%</span>
              </div>
              <div className="w-full bg-canvas-base h-2 border border-border-subtle">
                <div className="h-full bg-secondary transition-all" style={{ width: `${currentTelemetry.hamThr}%` }} />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-[10px] font-label-caps text-text-subtle mb-1">
                <span>BRAKE PRESSURE</span>
                <span className="font-data-mono-md text-text-primary font-bold">{currentTelemetry.hamBrk} BAR</span>
              </div>
              <div className="w-full bg-canvas-base h-2 border border-border-subtle">
                <div className="h-full bg-telemetry-amber transition-all" style={{ width: `${Math.min(100, (currentTelemetry.hamBrk / 120) * 100)}%` }} />
              </div>
            </div>

            <div className="flex justify-between items-center pt-2 border-t border-border-subtle text-xs">
              <span className="text-text-subtle font-label-caps text-[10px] uppercase">GEARBOX ENGAGED:</span>
              <span className="font-data-mono-md text-base text-text-primary font-bold">GEAR {currentTelemetry.hamGear}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Speed Delta Graph Visualization */}
      <div className="bg-canvas-base p-4 border border-border-subtle flex flex-col gap-2">
        <div className="flex justify-between items-center text-xs">
          <span className="font-label-caps text-[10px] uppercase text-text-muted font-bold">
            SPEED PROFILE TRACE ACROSS LAP (KM/H)
          </span>
          <div className="flex items-center gap-3 text-[10px] font-label-caps">
            <span className="flex items-center gap-1 text-primary-container">
              <span className="w-2 h-2 bg-primary-container inline-block"></span>
              Leclerc (#16)
            </span>
            <span className="flex items-center gap-1 text-telemetry-amber">
              <span className="w-2 h-2 bg-telemetry-amber inline-block"></span>
              Hamilton (#44)
            </span>
          </div>
        </div>

        {/* Speed Chart SVG */}
        <div className="w-full h-24 relative overflow-hidden">
          <svg viewBox="0 0 5793 400" preserveAspectRatio="none" className="w-full h-full">
            {/* Grid Lines */}
            <line x1="0" y1="100" x2="5793" y2="100" stroke="#292D33" strokeDasharray="10 10" />
            <line x1="0" y1="200" x2="5793" y2="200" stroke="#292D33" strokeDasharray="10 10" />
            <line x1="0" y1="300" x2="5793" y2="300" stroke="#292D33" strokeDasharray="10 10" />

            {/* Leclerc Speed Curve (Normalized: 400 - (speed / 360) * 360) */}
            <polyline
              fill="none"
              stroke="#E10600"
              strokeWidth="12"
              points={lapPoints.map((p) => `${p.dist},${400 - (p.lecSpeed / 360) * 380}`).join(' ')}
            />

            {/* Hamilton Speed Curve */}
            <polyline
              fill="none"
              stroke="#E7B45A"
              strokeWidth="10"
              strokeDasharray="25 15"
              points={lapPoints.map((p) => `${p.dist},${400 - (p.hamSpeed / 360) * 380}`).join(' ')}
            />

            {/* Scrubber vertical line */}
            <line
              x1={distanceMeters}
              y1="0"
              x2={distanceMeters}
              y2="400"
              stroke="#F4F4F2"
              strokeWidth="20"
              className="drop-shadow-[0_0_8px_#ffffff]"
            />
          </svg>
        </div>
      </div>
    </div>
  );
}
