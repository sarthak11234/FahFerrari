import React from 'react';

// Curated SVG track geometry paths for Formula 1 circuits
const circuitGeometries = {
  monza: 'M 60,160 L 320,160 C 345,160 355,150 340,130 L 305,90 C 295,75 280,80 265,85 L 205,100 C 190,105 180,90 185,75 L 200,45 C 205,30 190,20 175,25 L 130,40 C 110,45 100,65 110,85 L 140,125 C 150,140 135,148 120,148 L 70,148 C 50,148 45,158 60,160 Z',
  monaco: 'M 70,160 L 150,155 C 170,155 180,140 165,125 L 135,100 C 125,90 135,75 150,70 L 220,55 C 240,50 250,65 240,80 L 210,120 C 200,135 220,150 240,145 L 310,125 C 325,120 330,100 315,85 L 280,60 C 270,50 285,30 305,35 L 340,45 C 360,50 365,25 345,18 L 250,10 C 230,5 210,20 200,35 L 160,80 C 150,95 130,100 110,95 L 65,90 C 45,88 40,105 50,120 L 68,145 C 75,158 62,160 70,160 Z',
  bahrain: 'M 50,155 L 280,155 C 300,155 310,140 295,125 L 240,80 C 230,70 240,55 255,50 L 330,35 C 350,30 345,12 325,12 L 190,12 C 170,12 160,28 170,42 L 190,70 C 200,85 185,98 170,98 L 120,98 C 105,98 95,85 105,70 L 120,42 C 125,28 110,18 95,22 L 60,32 C 40,38 45,60 55,75 L 80,108 C 90,122 75,135 60,135 L 45,135 C 35,135 40,155 50,155 Z',
  marina_bay: 'M 65,155 L 240,155 C 260,155 270,140 255,125 L 210,90 C 200,80 210,65 225,60 L 310,42 C 330,38 335,18 315,18 L 220,18 C 200,18 190,32 200,48 L 215,70 C 225,85 210,98 195,98 L 140,98 C 125,98 115,85 125,70 L 140,42 C 145,28 130,18 115,22 L 75,32 C 55,38 60,60 70,75 L 95,108 C 105,122 90,135 75,135 L 60,135 C 50,135 52,155 65,155 Z',
  silverstone: 'M 60,155 L 195,155 C 225,155 245,138 235,118 L 215,90 C 205,75 215,60 235,55 L 305,42 C 330,38 335,18 315,15 L 245,15 C 215,15 195,30 205,50 L 215,68 C 220,82 200,92 185,88 L 125,72 C 105,68 95,82 100,98 L 110,120 C 115,135 100,142 85,142 L 58,142 C 45,142 45,155 60,155 Z',
  spa: 'M 50,165 L 180,165 C 200,165 220,150 210,132 L 190,105 C 180,90 190,75 210,70 L 320,42 C 345,38 340,15 315,15 L 190,15 C 160,15 150,32 160,50 L 175,78 C 185,95 165,110 145,105 L 90,92 C 70,88 60,102 65,120 L 75,145 C 80,160 65,165 50,165 Z',
};

// Generic procedural archetype fallback for other Grand Prix tracks
const fallbackGeometry = 'M 60,150 L 260,150 C 285,150 310,135 300,110 L 275,75 C 265,60 275,45 295,40 L 340,30 C 355,25 345,12 325,12 L 180,12 C 160,12 150,28 160,45 L 175,75 C 185,92 165,108 145,102 L 95,90 C 75,85 65,100 70,118 L 80,138 C 85,148 70,150 60,150 Z';

export default function CircuitTrackSvg({ circuitId, circuitName }) {
  const pathData = circuitGeometries[circuitId] || fallbackGeometry;

  return (
    <div className="w-full bg-canvas-base border border-border-subtle p-3 flex flex-col items-center justify-center relative overflow-hidden select-none">
      {/* Background Grid Pattern */}
      <div
        className="absolute inset-0 opacity-10 pointer-events-none"
        style={{
          backgroundImage: 'radial-gradient(#55C58A 1px, transparent 1px)',
          backgroundSize: '16px 16px',
        }}
      />

      {/* SVG Circuit Canvas */}
      <svg
        viewBox="0 0 380 180"
        className="w-full h-36 max-w-[340px] overflow-visible drop-shadow-[0_0_12px_rgba(225,6,0,0.25)]"
      >
        {/* Track Glow Halo */}
        <path
          d={pathData}
          fill="none"
          stroke="#E10600"
          strokeWidth="6"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="opacity-25 blur-[2px]"
        />

        {/* Main Track Line */}
        <path
          d={pathData}
          fill="none"
          stroke="#E10600"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Start/Finish Line Indicator */}
        <line
          x1="60"
          y1="148"
          x2="60"
          y2="172"
          stroke="#F4F4F2"
          strokeWidth="3"
          strokeLinecap="square"
        />

        {/* Animated Beacon Dot Tracing the Track */}
        <circle r="4" fill="#55C58A" className="shadow-[0_0_8px_#55C58A]">
          <animateMotion
            dur="6s"
            repeatCount="indefinite"
            path={pathData}
          />
        </circle>
      </svg>

      {/* Track Overlay HUD Tags */}
      <div className="w-full flex justify-between items-center text-[10px] font-mono text-text-subtle pt-2 border-t border-border-subtle/50 mt-1">
        <span className="flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-secondary animate-pulse" />
          <span>START / FINISH LINE</span>
        </span>
        <span className="text-primary-container font-bold uppercase font-label-caps tracking-wider">
          FIA GRADE-1 HOMOLOGATION
        </span>
      </div>
    </div>
  );
}

