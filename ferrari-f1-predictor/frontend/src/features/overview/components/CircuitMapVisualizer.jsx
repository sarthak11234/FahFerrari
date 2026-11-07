import React, { useState } from 'react';
import { MapPin, Zap, Gauge, Flag, Activity } from 'lucide-react';

export default function CircuitMapVisualizer({ selectedCircuitId = 'monza' }) {
  const [activeCircuit, setActiveCircuit] = useState(selectedCircuitId);

  const circuits = {
    monza: {
      name: 'Autodromo Nazionale Monza',
      country: 'Italy',
      type: 'Temple of Speed / Low Drag',
      length: '5.793 km',
      turns: 11,
      topSpeed: '354.2 km/h',
      winProb: '33.4%',
      expPoints: '24.1 pts',
      svgPath: 'M 60,180 L 320,180 C 340,180 350,170 340,150 L 310,100 C 300,85 285,90 270,95 L 210,110 C 195,115 185,100 190,85 L 205,50 C 210,35 195,25 180,30 L 130,45 C 110,50 100,70 110,90 L 140,140 C 150,155 135,165 120,165 L 70,165 C 50,165 45,175 60,180 Z',
      sectors: [
        { name: 'Sector 1 (Variante del Rettifilo)', speed: '352 km/h' },
        { name: 'Sector 2 (Curva di Lesmo & Roggia)', speed: '264 km/h' },
        { name: 'Sector 3 (Variante Ascari & Parabolica)', speed: '338 km/h' },
      ],
      ferrariAdvantage: 'High-speed straight-line efficiency and aggressive low-downforce beam wing deployment give Ferrari an edge against McLaren.',
    },
    monaco: {
      name: 'Circuit de Monaco',
      country: 'Monaco',
      type: 'Maximum Downforce / Street',
      length: '3.337 km',
      turns: 19,
      topSpeed: '292.4 km/h',
      winProb: '31.2%',
      expPoints: '23.4 pts',
      svgPath: 'M 80,180 L 160,175 C 180,175 190,160 175,145 L 140,115 C 130,105 140,90 155,85 L 230,70 C 250,65 260,80 250,95 L 220,140 C 210,155 230,170 250,165 L 320,145 C 335,140 340,120 325,105 L 290,75 C 280,65 295,45 315,50 L 350,60 C 370,65 375,40 355,30 L 260,15 C 240,10 220,25 210,40 L 170,90 C 160,105 140,110 120,105 L 70,100 C 50,98 45,115 55,130 L 75,160 C 85,175 70,180 80,180 Z',
      sectors: [
        { name: 'Sector 1 (Sainte Devote & Beau Rivage)', speed: '286 km/h' },
        { name: 'Sector 2 (Mirabeau & Grand Hotel Hairpin)', speed: '48 km/h (Slowest apex)' },
        { name: 'Sector 3 (Swimming Pool & La Rascasse)', speed: '215 km/h' },
      ],
      ferrariAdvantage: 'Mechanical grip, short-wheelbase turn-in, and kerb compliance deliver Leclerc peak pole position conversion probability.',
    },
    bahrain: {
      name: 'Bahrain International Circuit (Sakhir)',
      country: 'Bahrain',
      type: 'Power & Thermal Traction',
      length: '5.412 km',
      turns: 15,
      topSpeed: '338.5 km/h',
      winProb: '18.9%',
      expPoints: '18.1 pts',
      svgPath: 'M 50,170 L 280,170 C 300,170 310,155 295,140 L 240,90 C 230,80 240,65 255,60 L 330,45 C 350,40 345,20 325,20 L 190,20 C 170,20 160,35 170,50 L 190,80 C 200,95 185,110 170,110 L 120,110 C 105,110 95,95 105,80 L 120,50 C 125,35 110,25 95,30 L 60,40 C 40,45 45,70 55,85 L 80,120 C 90,135 75,150 60,150 L 45,150 C 35,150 40,170 50,170 Z',
      sectors: [
        { name: 'Sector 1 (Main Straight to T1 Hairpin)', speed: '334 km/h' },
        { name: 'Sector 2 (High-Speed Sweepers T5-T7)', speed: '258 km/h' },
        { name: 'Sector 3 (Turn 10 Off-Camber Braking)', speed: '312 km/h' },
      ],
      ferrariAdvantage: 'High turbo boost response out of Turn 1 and Turn 4 hairpins, though high asphalt abrasiveness taxes rear tire life.',
    },
    marina_bay: {
      name: 'Marina Bay Street Circuit',
      country: 'Singapore',
      type: 'Night Street / High Thermal Load',
      length: '4.940 km',
      turns: 19,
      topSpeed: '308.2 km/h',
      winProb: '28.5%',
      expPoints: '21.5 pts',
      svgPath: 'M 70,175 L 240,175 C 260,175 270,160 255,145 L 210,105 C 200,95 210,80 225,75 L 310,55 C 330,50 335,30 315,30 L 220,30 C 200,30 190,45 200,60 L 215,85 C 225,100 210,115 195,115 L 140,115 C 125,115 115,100 125,85 L 140,55 C 145,40 130,30 115,35 L 75,45 C 55,50 60,75 70,90 L 95,125 C 105,140 90,155 75,155 L 60,155 C 50,155 55,175 70,175 Z',
      sectors: [
        { name: 'Sector 1 (Sheares Turn 1-3 Complex)', speed: '298 km/h' },
        { name: 'Sector 2 (Anderson Bridge & Padang)', speed: '192 km/h' },
        { name: 'Sector 3 (Floating Platform Chicanes)', speed: '210 km/h' },
      ],
      ferrariAdvantage: 'Night ambient temperatures reduce thermal blistering, allowing aggressive 1-stop strategy and high downforce pace.',
    },
  };

  const curr = circuits[activeCircuit] || circuits.monza;

  return (
    <div className="bg-surface-panel p-6 border border-border-subtle flex flex-col gap-6">
      {/* Circuit Selector Tabs */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-border-subtle pb-4">
        <div>
          <div className="flex items-center gap-2">
            <MapPin className="w-4 h-4 text-primary-container" />
            <span className="font-label-caps text-xs uppercase text-primary-container tracking-wider font-bold">
              RACETRACK TELEMETRY ENGINE
            </span>
          </div>
          <h3 className="font-headline-md text-2xl uppercase text-text-primary font-bold mt-1">
            CIRCUIT ARCHETYPE &amp; SPEED TRAP MAPS
          </h3>
        </div>

        <div className="flex flex-wrap items-center gap-1 bg-surface-raised p-1 border border-border-subtle">
          {Object.entries(circuits).map(([id, c]) => (
            <button
              key={id}
              onClick={() => setActiveCircuit(id)}
              className={`px-3 py-1.5 text-xs font-label-caps uppercase transition-colors ${
                activeCircuit === id
                  ? 'bg-primary-container text-text-primary font-bold shadow-[0_0_10px_rgba(225,6,0,0.4)]'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              {c.country}
            </button>
          ))}
        </div>
      </div>

      {/* Main Visualizer Content */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
        {/* SVG Circuit Vector Map with Animated Lap Pulse */}
        <div className="lg:col-span-6 bg-canvas-base border border-border-subtle p-6 flex flex-col items-center justify-center relative overflow-hidden min-h-[300px]">
          {/* Subtle Grid Lines in Background */}
          <div className="absolute inset-0 bg-[linear-gradient(to_right,#191C21_1px,transparent_1px),linear-gradient(to_bottom,#191C21_1px,transparent_1px)] bg-[size:24px_24px] opacity-40 pointer-events-none" />

          {/* Circuit Label HUD */}
          <div className="absolute top-3 left-3 text-[10px] font-label-caps uppercase text-text-subtle flex flex-col">
            <span className="text-primary-container font-bold font-data-mono-md">{curr.name}</span>
            <span>{curr.type}</span>
          </div>

          <div className="absolute bottom-3 right-3 text-[10px] font-label-caps uppercase text-secondary flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-secondary animate-ping status-dot"></span>
            <span>ACTIVE SF-27 TELEMETRY MAPPING</span>
          </div>

          {/* SVG Map */}
          <svg
            viewBox="0 0 400 220"
            className="w-full max-w-[420px] h-[220px] overflow-visible drop-shadow-[0_0_15px_rgba(225,6,0,0.3)]"
          >
            {/* Background Glow Path */}
            <path
              d={curr.svgPath}
              fill="none"
              stroke="#E10600"
              strokeWidth="10"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="opacity-20"
            />
            {/* Primary Track Line */}
            <path
              d={curr.svgPath}
              fill="none"
              stroke="#F4F4F2"
              strokeWidth="4"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            {/* Speed Flow Dash Animation */}
            <path
              d={curr.svgPath}
              fill="none"
              stroke="#E10600"
              strokeWidth="4"
              strokeDasharray="24 160"
              className="animate-pulse"
              strokeLinecap="round"
            />
            {/* Finish Line Marker */}
            <circle cx="60" cy="180" r="5" fill="#55C58A" className="shadow-[0_0_8px_#55C58A]" />
          </svg>
        </div>

        {/* Circuit Telemetry Analytics */}
        <div className="lg:col-span-6 flex flex-col justify-between gap-4">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-surface-raised p-3 border border-border-subtle text-xs">
            <div>
              <span className="font-label-caps text-[10px] text-text-subtle uppercase block">TRACK LENGTH</span>
              <span className="font-data-mono-md text-base text-text-primary font-bold tnum">{curr.length}</span>
            </div>
            <div>
              <span className="font-label-caps text-[10px] text-text-subtle uppercase block">CORNER COUNT</span>
              <span className="font-data-mono-md text-base text-text-primary font-bold tnum">{curr.turns} Turns</span>
            </div>
            <div>
              <span className="font-label-caps text-[10px] text-text-subtle uppercase block">PEAK SPEED</span>
              <span className="font-data-mono-md text-base text-secondary font-bold tnum">{curr.topSpeed}</span>
            </div>
            <div>
              <span className="font-label-caps text-[10px] text-text-subtle uppercase block">P(WIN) PROB</span>
              <span className="font-data-mono-md text-base text-primary-container font-bold tnum">{curr.winProb}</span>
            </div>
          </div>

          {/* Sector Speed Breakdown */}
          <div className="flex flex-col gap-2">
            <span className="font-label-caps text-xs text-text-muted uppercase font-semibold">
              SECTOR TELEMETRY SPEED TRAPS
            </span>
            <div className="flex flex-col gap-1.5">
              {curr.sectors.map((sec, idx) => (
                <div key={idx} className="flex justify-between items-center text-xs bg-canvas-base px-3 py-2 border border-border-subtle/70">
                  <span className="text-text-muted">{sec.name}</span>
                  <span className="font-data-mono-md font-bold text-text-primary tnum">{sec.speed}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Scuderia Strategy Insight */}
          <div className="p-3 bg-primary-container/10 border-l-2 border-primary-container text-xs text-text-muted leading-relaxed">
            <span className="font-label-caps uppercase text-primary-container font-bold block mb-0.5">
              SCUDERIA SF-27 TACTICAL ADVANTAGE:
            </span>
            {curr.ferrariAdvantage}
          </div>
        </div>
      </div>
    </div>
  );
}

