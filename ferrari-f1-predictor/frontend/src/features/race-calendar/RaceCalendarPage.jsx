import React, { useState, useEffect } from 'react';
import { get2027Predictions } from '../../services/api';
import CredibleIntervalBar from '../../components/common/CredibleIntervalBar';
import CircuitTrackSvg from '../../components/common/CircuitTrackSvg';
import frozenForecast2027 from '../../data/preseason2027Data.json';
import {
  Calendar,
  Flag,
  MapPin,
  Clock,
  Zap,
  Award,
  AlertTriangle,
  ChevronRight,
  Filter,
  Layers,
  X,
} from 'lucide-react';

export default function RaceCalendarPage() {
  const [data, setData] = useState(frozenForecast2027);
  const [loading, setLoading] = useState(false);
  const [selectedRace, setSelectedRace] = useState(frozenForecast2027.race_by_race_forecast[0]);
  const [filterType, setFilterType] = useState('all'); // 'all' | 'sprint' | 'high_win'

  // Race calendar dates and countries mapping
  const raceDetails = {
    bahrain: { country: 'Bahrain', date: 'FEB 28 – MAR 02', laps: 57, length: '5.412 km', archetype: 'Power & Traction' },
    jeddah: { country: 'Saudi Arabia', date: 'MAR 07 – MAR 09', laps: 50, length: '6.174 km', archetype: 'High-Speed Street' },
    albert_park: { country: 'Australia', date: 'MAR 21 – MAR 23', laps: 58, length: '5.278 km', archetype: 'Street / Semi-Permanent' },
    suzuka: { country: 'Japan', date: 'APR 04 – APR 06', laps: 53, length: '5.807 km', archetype: 'Lateral Aero High-G' },
    shanghai: { country: 'China', date: 'APR 18 – APR 20', laps: 56, length: '5.451 km', archetype: 'Front-Limited Traction' },
    miami: { country: 'United States', date: 'MAY 02 – MAY 04', laps: 57, length: '5.412 km', archetype: 'Street Hybrid' },
    imola: { country: 'Italy', date: 'MAY 16 – MAY 18', laps: 63, length: '4.909 km', archetype: 'Technical Curbs' },
    monaco: { country: 'Monaco', date: 'MAY 23 – MAY 25', laps: 78, length: '3.337 km', archetype: 'Max Downforce Street' },
    montreal: { country: 'Canada', date: 'JUN 13 – JUN 15', laps: 70, length: '4.361 km', archetype: 'Stop-and-Go Traction' },
    barcelona: { country: 'Spain', date: 'JUN 27 – JUN 29', laps: 66, length: '4.657 km', archetype: 'High-Speed Aerodynamic' },
    red_bull_ring: { country: 'Austria', date: 'JUL 04 – JUL 06', laps: 71, length: '4.318 km', archetype: 'Short Lap Power' },
    silverstone: { country: 'Great Britain', date: 'JUL 11 – JUL 13', laps: 52, length: '5.891 km', archetype: 'High Lateral High-Speed' },
    hungaroring: { country: 'Hungary', date: 'AUG 01 – AUG 03', laps: 70, length: '4.381 km', archetype: 'Maximum Downforce' },
    spa: { country: 'Belgium', date: 'AUG 29 – AUG 31', laps: 44, length: '7.004 km', archetype: 'Low Drag / High Elevation' },
    zandvoort: { country: 'Netherlands', date: 'SEP 05 – SEP 07', laps: 72, length: '4.259 km', archetype: 'Banked Downforce' },
    monza: { country: 'Italy', date: 'SEP 12 – SEP 14', laps: 53, length: '5.793 km', archetype: 'Temple of Speed / Low Drag' },
    baku: { country: 'Azerbaijan', date: 'SEP 26 – SEP 28', laps: 51, length: '6.003 km', archetype: 'High Speed Long Straight Street' },
    marina_bay: { country: 'Singapore', date: 'OCT 03 – OCT 05', laps: 62, length: '4.940 km', archetype: 'Night Street / High Thermal' },
    cota: { country: 'United States', date: 'OCT 22 – OCT 24', laps: 56, length: '5.513 km', archetype: 'Bumpy Sector 1 S-Curves' },
    mexico: { country: 'Mexico', date: 'OCT 29 – OCT 31', laps: 71, length: '4.304 km', archetype: 'Thin Air / Cooling Limited' },
    interlagos: { country: 'Brazil', date: 'NOV 12 – NOV 14', laps: 71, length: '4.309 km', archetype: 'Anti-Clockwise Rollercoaster' },
    las_vegas: { country: 'United States', date: 'NOV 26 – NOV 28', laps: 50, length: '6.201 km', archetype: 'Cold Night Low Drag' },
    lusail: { country: 'Qatar', date: 'DEC 03 – DEC 05', laps: 57, length: '5.419 km', archetype: 'High Lateral Tire Scrub' },
    yas_marina: { country: 'Abu Dhabi', date: 'DEC 10 – DEC 12', laps: 58, length: '5.281 km', archetype: 'Season Finale Twighlight' },
  };

  useEffect(() => {
    async function load() {
      try {
        const preds = await get2027Predictions();
        setData(preds);
        if (preds.race_by_race_forecast && preds.race_by_race_forecast.length > 0) {
          setSelectedRace(preds.race_by_race_forecast[0]);
        }
      } catch (err) {
        console.error('Failed to load race calendar:', err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading || !data) {
    return (
      <div className="w-full min-h-[60vh] flex flex-col items-center justify-center gap-3">
        <div className="w-8 h-8 border-2 border-primary-container border-t-transparent animate-spin"></div>
        <span className="font-label-caps text-xs uppercase text-text-muted">Loading 24-Round Master Race Calendar...</span>
      </div>
    );
  }

  const races = data.race_by_race_forecast;

  const filteredRaces = races.filter((r) => {
    if (filterType === 'sprint') return r.is_sprint;
    if (filterType === 'high_win') return r.win_probability >= 0.20;
    return true;
  });

  const getExtraInfo = (circuitId) => {
    return raceDetails[circuitId] || {
      country: 'FIA Grand Prix',
      date: 'TBA 2027',
      laps: 55,
      length: '5.0 km',
      archetype: 'Standard Circuit',
    };
  };

  return (
    <div className="w-full flex flex-col">
      {/* Header Banner */}
      <section className="w-full bg-surface-panel border-b border-border-subtle py-8">
        <div className="max-w-[1440px] mx-auto px-4 md:px-margin-desktop flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <span className="w-2 h-4 bg-primary-container"></span>
            <span className="font-label-caps text-xs uppercase text-primary-container tracking-widest font-bold">
              SCUDERIA COMMAND · CALENDAR SIMULATION
            </span>
          </div>
          <h1 className="font-headline-lg text-headline-lg uppercase text-text-primary font-bold">
            THE ROAD TO THE CHAMPIONSHIP
          </h1>
          <p className="font-body-md text-text-muted max-w-3xl leading-relaxed">
            The complete 24-round 2027 Formula 1 calendar with circuit telemetry profiles, sprint flags, simulated Ferrari points intervals, win probabilities, and discrete Weibull component hazard risks.
          </p>
        </div>
      </section>

      <div className="max-w-[1440px] mx-auto px-4 md:px-margin-desktop py-12 flex flex-col gap-12">
        {/* Controls and Summary Strip */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 bg-surface-panel p-4 border border-border-subtle">
          <div className="flex flex-wrap items-center gap-3 text-xs font-label-caps uppercase">
            <span className="text-text-muted">FILTER ROUNDS:</span>
            <button
              onClick={() => setFilterType('all')}
              className={`px-3 py-1 transition-colors ${
                filterType === 'all' ? 'bg-primary-container text-text-primary font-bold' : 'bg-surface-raised text-text-muted hover:text-text-primary border border-border-subtle'
              }`}
            >
              All 24 Grands Prix
            </button>
            <button
              onClick={() => setFilterType('sprint')}
              className={`px-3 py-1 transition-colors ${
                filterType === 'sprint' ? 'bg-primary-container text-text-primary font-bold' : 'bg-surface-raised text-text-muted hover:text-text-primary border border-border-subtle'
              }`}
            >
              10 Sprint Events
            </button>
            <button
              onClick={() => setFilterType('high_win')}
              className={`px-3 py-1 transition-colors ${
                filterType === 'high_win' ? 'bg-primary-container text-text-primary font-bold' : 'bg-surface-raised text-text-muted hover:text-text-primary border border-border-subtle'
              }`}
            >
              High Win Probability (&ge;20%)
            </button>
          </div>

          <div className="text-xs font-label-caps text-text-subtle uppercase">
            SHOWING {filteredRaces.length} OF 24 EVENTS
          </div>
        </div>

        {/* Calendar Grid + Detailed Sidebar layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
          {/* Main 2-column cards list */}
          <div className="lg:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-4">
            {filteredRaces.map((r) => {
              const extra = getExtraInfo(r.circuit_id);
              const isSelected = selectedRace?.round === r.round;
              return (
                <div
                  key={r.round}
                  onClick={() => setSelectedRace(r)}
                  className={`bg-surface-panel p-5 border cursor-pointer transition-all flex flex-col justify-between gap-4 ${
                    isSelected
                      ? 'border-primary-container bg-surface-raised shadow-[0_0_15px_rgba(225,6,0,0.15)]'
                      : 'border-border-subtle hover:border-border-focus'
                  }`}
                >
                  <div>
                    <div className="flex justify-between items-start text-xs font-label-caps">
                      <div className="flex items-center gap-1.5">
                        <span className="font-data-mono-md font-bold text-primary-container">
                          ROUND {r.round.toString().padStart(2, '0')}
                        </span>
                        <span className="text-text-subtle">· {extra.country}</span>
                      </div>
                      {r.is_sprint && (
                        <span className="px-1.5 py-0.5 bg-telemetry-amber/20 border border-telemetry-amber/40 text-telemetry-amber text-[10px] font-bold">
                          SPRINT
                        </span>
                      )}
                    </div>

                    <h3 className="font-headline-sm text-lg uppercase text-text-primary font-bold mt-1.5 leading-snug">
                      {r.circuit}
                    </h3>
                    <div className="flex items-center gap-3 text-xs text-text-subtle mt-1 font-body-sm">
                      <span className="flex items-center gap-1">
                        <Calendar className="w-3.5 h-3.5" />
                        {extra.date}
                      </span>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2 bg-surface-raised p-2 text-xs border border-border-subtle">
                    <div>
                      <span className="font-label-caps text-[10px] text-text-subtle uppercase block">EXP FERRARI PTS</span>
                      <span className="font-data-mono-md text-base text-text-primary font-bold tnum">
                        {r.expected_points.toFixed(1)}
                      </span>
                    </div>
                    <div>
                      <span className="font-label-caps text-[10px] text-text-subtle uppercase block">P(WIN)</span>
                      <span className="font-data-mono-md text-base text-primary-container font-bold tnum">
                        {(r.win_probability * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  <div className="flex justify-between items-center text-xs text-text-subtle pt-2 border-t border-border-subtle">
                    <span>Podium: {(r.podium_probability * 100).toFixed(1)}%</span>
                    <span className="text-telemetry-amber">DNF Hazard: {(r.dnf_risk * 100).toFixed(1)}%</span>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Right Sidebar: Selected Race Telemetry Inspector */}
          {selectedRace && (
            <div className="bg-surface-panel p-6 border border-border-focus sticky top-24 flex flex-col gap-6">
              <div className="flex justify-between items-start border-b border-border-subtle pb-4">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-data-mono-md text-primary-container text-xs font-bold uppercase">
                      ROUND {selectedRace.round.toString().padStart(2, '0')} TELEMETRY INSPECTOR
                    </span>
                    {selectedRace.is_sprint && (
                      <span className="px-1.5 py-0.5 bg-telemetry-amber/20 text-telemetry-amber font-label-caps text-[10px] font-bold">
                        SPRINT WEEKEND
                      </span>
                    )}
                  </div>
                  <h3 className="font-headline-md text-2xl uppercase text-text-primary font-bold mt-1">
                    {selectedRace.circuit}
                  </h3>
                  <span className="text-xs text-text-muted font-body-sm block mt-0.5">
                    {getExtraInfo(selectedRace.circuit_id).country} · {getExtraInfo(selectedRace.circuit_id).date}
                  </span>
                </div>
              </div>

              {/* Circuit Parameters */}
              <div className="grid grid-cols-2 gap-3 text-xs bg-surface-raised p-3 border border-border-subtle">
                <div>
                  <span className="font-label-caps text-[10px] text-text-subtle uppercase block">AERO ARCHETYPE</span>
                  <span className="font-body-sm text-text-primary font-bold block mt-0.5">
                    {getExtraInfo(selectedRace.circuit_id).archetype}
                  </span>
                </div>
                <div>
                  <span className="font-label-caps text-[10px] text-text-subtle uppercase block">RACE DISTANCE</span>
                  <span className="font-body-sm text-text-primary font-bold block mt-0.5">
                    {getExtraInfo(selectedRace.circuit_id).laps} Laps ({getExtraInfo(selectedRace.circuit_id).length})
                  </span>
                </div>
              </div>

              {/* Interactive Circuit Track Map Visualizer */}
              <CircuitTrackSvg circuitId={selectedRace.circuit_id} circuitName={selectedRace.circuit} />

              {/* Simulation Forecast Metrics */}
              <div className="flex flex-col gap-3">
                <span className="font-label-caps text-xs text-text-muted uppercase font-semibold">
                  SIMULATED PROBABILISTIC OUTCOMES
                </span>

                <div className="flex justify-between items-center py-2 border-b border-border-subtle/50 text-xs">
                  <span className="text-text-muted">Expected Ferrari Team Points:</span>
                  <span className="font-data-mono-md text-text-primary font-bold text-base tnum">
                    {selectedRace.expected_points.toFixed(1)} pts
                  </span>
                </div>

                <div className="flex justify-between items-center py-2 border-b border-border-subtle/50 text-xs">
                  <span className="text-text-muted">90% HDI Points Range:</span>
                  <span className="font-data-mono-md text-text-primary font-bold tnum">
                    [{selectedRace.points_ci_90[0].toFixed(0)} – {selectedRace.points_ci_90[1].toFixed(0)} pts]
                  </span>
                </div>

                <div className="flex justify-between items-center py-2 border-b border-border-subtle/50 text-xs">
                  <span className="text-text-muted">Ferrari Win Probability:</span>
                  <span className="font-data-mono-md text-primary-container font-bold text-base tnum">
                    {(selectedRace.win_probability * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="flex justify-between items-center py-2 border-b border-border-subtle/50 text-xs">
                  <span className="text-text-muted">Podium Finish Probability:</span>
                  <span className="font-data-mono-md text-secondary font-bold text-base tnum">
                    {(selectedRace.podium_probability * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="flex justify-between items-center py-2 text-xs">
                  <span className="text-text-muted">Weibull DNF Hazard Risk:</span>
                  <span className="font-data-mono-md text-telemetry-amber font-bold text-base tnum">
                    {(selectedRace.dnf_risk * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Strategy & Tactical Note */}
              <div className="bg-surface-raised p-4 border border-border-subtle text-xs text-text-muted leading-relaxed">
                <span className="font-label-caps uppercase text-text-primary font-bold block mb-1">
                  TACTICAL PIT STRATEGY FORECAST
                </span>
                Under simulated 2027 degradation dynamics, {selectedRace.circuit} features an optimal 1-stop strategy window (Laps 19–24 Medium &rarr; Hard). Mechanical safety car probability is estimated at {selectedRace.circuit_id === 'monaco' || selectedRace.circuit_id === 'marina_bay' ? '68.2%' : '34.5%'}.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

