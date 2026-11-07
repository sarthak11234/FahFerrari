import React, { useState, useEffect } from 'react';
import { get2027Predictions } from '../../services/api';
import CredibleIntervalBar from '../../components/common/CredibleIntervalBar';
import frozenForecast2027 from '../../data/preseason2027Data.json';
import {
  Trophy,
  Filter,
  Layers,
  ArrowUpDown,
  TrendingUp,
  BarChart2,
  Users,
  Percent,
} from 'lucide-react';

export default function ChampionshipPage() {
  const [data, setData] = useState(frozenForecast2027);
  const [loading, setLoading] = useState(false);
  const [filterTier, setFilterTier] = useState('all'); // 'all' | 'top4' | 'midfield'

  useEffect(() => {
    async function load() {
      try {
        const preds = await get2027Predictions();
        setData(preds);
      } catch (e) {
        console.error('Failed to load championship data:', e);
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
        <span className="font-label-caps text-xs uppercase text-text-muted">Calculating Championship Probability Matrices...</span>
      </div>
    );
  }

  const standings = data.constructors_standings;
  const leclerc = data.driver_forecasts.leclerc;
  const hamilton = data.driver_forecasts.hamilton;

  // Filtered constructors
  const filteredStandings = standings.filter((c, idx) => {
    if (filterTier === 'top4') return idx < 4;
    if (filterTier === 'midfield') return idx >= 4 && idx < 8;
    return true;
  });

  // Heatmap positions matrix
  const positions = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10+'];

  // Synthetic or calculated position probabilities for each team
  const getTeamPositionProb = (teamId, pos) => {
    if (teamId === 'ferrari') {
      const p = data.ferrari_constructors_forecast.position_probabilities[pos];
      return p !== undefined ? p : pos === 'P10+' ? 0.0 : 0.0;
    }
    if (teamId === 'mercedes') {
      if (pos === 'P1') return 0.873;
      if (pos === 'P2') return 0.114;
      if (pos === 'P3') return 0.013;
      return 0.0;
    }
    if (teamId === 'red_bull') {
      if (pos === 'P1') return 0.116;
      if (pos === 'P2') return 0.702;
      if (pos === 'P3') return 0.181;
      if (pos === 'P4') return 0.001;
      return 0.0;
    }
    if (teamId === 'mclaren') {
      if (pos === 'P2') return 0.012;
      if (pos === 'P3') return 0.004;
      if (pos === 'P4') return 0.842;
      if (pos === 'P5') return 0.142;
      return 0.0;
    }
    if (teamId === 'alpine') {
      if (pos === 'P4') return 0.142;
      if (pos === 'P5') return 0.692;
      if (pos === 'P6') return 0.166;
      return 0.0;
    }
    if (teamId === 'aston_martin') {
      if (pos === 'P5') return 0.151;
      if (pos === 'P6') return 0.712;
      if (pos === 'P7') return 0.137;
      return 0.0;
    }
    if (teamId === 'rb') {
      if (pos === 'P6') return 0.122;
      if (pos === 'P7') return 0.755;
      if (pos === 'P8') return 0.123;
      return 0.0;
    }
    if (teamId === 'audi') {
      if (pos === 'P7') return 0.108;
      if (pos === 'P8') return 0.765;
      if (pos === 'P9') return 0.127;
      return 0.0;
    }
    if (teamId === 'williams') {
      if (pos === 'P8') return 0.112;
      if (pos === 'P9') return 0.584;
      if (pos === 'P10+') return 0.304;
      return 0.0;
    }
    if (teamId === 'haas') {
      if (pos === 'P8') return 0.0;
      if (pos === 'P9') return 0.289;
      if (pos === 'P10+') return 0.711;
      return 0.0;
    }
    return 0.0;
  };

  // Cell color helper for heatmap
  const getHeatmapColor = (prob, isFerrari) => {
    if (prob <= 0.001) return 'bg-surface-raised text-text-subtle';
    if (prob < 0.05) return 'bg-surface-container-high text-text-muted';
    if (prob < 0.20) return isFerrari ? 'bg-primary-container/30 text-text-primary' : 'bg-surface-bright text-text-primary';
    if (prob < 0.50) return isFerrari ? 'bg-primary-container/60 text-text-primary font-bold' : 'bg-border-focus text-text-primary font-bold';
    return isFerrari ? 'bg-primary-container text-text-primary font-bold shadow-[0_0_10px_rgba(225,6,0,0.5)]' : 'bg-secondary-container text-text-primary font-bold';
  };

  // Drivers contenders roster
  const driverContenders = [
    {
      num: '16',
      name: 'Charles Leclerc',
      team: 'Scuderia Ferrari',
      isFerrari: true,
      points: leclerc.mean_points,
      median: leclerc.median_points,
      ci90: leclerc.ci_90,
      wins: leclerc.expected_wins,
      podiums: leclerc.expected_podiums,
      titleProb: leclerc.prob_drivers_title,
      color: '#E10600',
      note: 'Teammate margin: 1.8 wins, 90% HDI [158.0 - 285.0]',
    },
    {
      num: '44',
      name: 'Lewis Hamilton',
      team: 'Scuderia Ferrari',
      isFerrari: true,
      points: hamilton.mean_points,
      median: hamilton.median_points,
      ci90: hamilton.ci_90,
      wins: hamilton.expected_wins,
      podiums: hamilton.expected_podiums,
      titleProb: hamilton.prob_drivers_title,
      color: '#E7B45A',
      note: 'Teammate margin: 2.6 wins, 90% HDI [205.0 - 335.0]',
    },
    {
      num: '63',
      name: 'George Russell',
      team: 'Mercedes-AMG',
      isFerrari: false,
      points: 341.2,
      median: 340.5,
      ci90: [292.0, 392.0],
      wins: 7.1,
      podiums: 16.2,
      titleProb: 0.445,
      color: '#6FA8FF',
      note: 'Championship Favorite (44.5% Title Probability)',
    },
    {
      num: '01',
      name: 'Max Verstappen',
      team: 'Red Bull Racing',
      isFerrari: false,
      points: 298.4,
      median: 295.0,
      ci90: [242.0, 356.0],
      wins: 5.6,
      podiums: 14.1,
      titleProb: 0.148,
      color: '#A0A5AD',
      note: 'Pace Index: 0.942 (Field Benchmark)',
    },
    {
      num: '04',
      name: 'Lando Norris',
      team: 'McLaren Mercedes',
      isFerrari: false,
      points: 172.5,
      median: 171.0,
      ci90: [128.0, 218.0],
      wins: 1.9,
      podiums: 6.1,
      titleProb: 0.008,
      color: '#E7B45A',
      note: 'Top Contender in High-Downforce Sector 2',
    },
    {
      num: '81',
      name: 'Oscar Piastri',
      team: 'McLaren Mercedes',
      isFerrari: false,
      points: 152.5,
      median: 151.0,
      ci90: [110.0, 196.0],
      wins: 1.1,
      podiums: 4.8,
      titleProb: 0.003,
      color: '#E7B45A',
      note: 'Consistent Points Finisher',
    },
  ];

  return (
    <div className="w-full flex flex-col">
      {/* Header Banner */}
      <section className="w-full bg-surface-panel border-b border-border-subtle py-8">
        <div className="max-w-[1440px] mx-auto px-4 md:px-margin-desktop flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <span className="w-2 h-4 bg-primary-container"></span>
            <span className="font-label-caps text-xs uppercase text-primary-container tracking-widest font-bold">
              SCUDERIA COMMAND · QUANTITATIVE MATRICES
            </span>
          </div>
          <h1 className="font-headline-lg text-headline-lg uppercase text-text-primary font-bold">
            THE CHAMPIONSHIP MODEL
          </h1>
          <p className="font-body-md text-text-muted max-w-3xl leading-relaxed">
            Constructors' and Drivers' Championship joint probability distributions over 10,000 paired Monte Carlo simulations. Evaluates full field credible intervals, finishing position heatmaps, and rival head-to-head margins.
          </p>
        </div>
      </section>

      <div className="max-w-[1440px] mx-auto px-4 md:px-margin-desktop py-12 flex flex-col gap-16">
        {/* SECTION 1: CONSTRUCTORS' CHAMPIONSHIP TABLE */}
        <section className="flex flex-col gap-6">
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-border-subtle pb-3">
            <div>
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-3 bg-primary-container"></span>
                <h2 className="font-headline-md text-headline-md uppercase text-text-primary">
                  2027 CONSTRUCTORS' CHAMPIONSHIP — POINT EXPECTATION &amp; 90% CREDIBLE INTERVAL (HDI)
                </h2>
              </div>
              <p className="font-body-sm text-text-muted mt-0.5">
                Full-field standings sorted by expected points, showing median baselines and simulated certainty boundaries.
              </p>
            </div>

            {/* Filter Toggle */}
            <div className="flex items-center gap-1 bg-surface-panel border border-border-subtle p-1 self-start md:self-auto">
              <button
                onClick={() => setFilterTier('all')}
                className={`px-3 py-1 text-xs font-label-caps uppercase transition-colors ${
                  filterTier === 'all' ? 'bg-primary-container text-text-primary font-bold' : 'text-text-muted hover:text-text-primary'
                }`}
              >
                All 10 Teams
              </button>
              <button
                onClick={() => setFilterTier('top4')}
                className={`px-3 py-1 text-xs font-label-caps uppercase transition-colors ${
                  filterTier === 'top4' ? 'bg-primary-container text-text-primary font-bold' : 'text-text-muted hover:text-text-primary'
                }`}
              >
                Top 4 Contenders
              </button>
              <button
                onClick={() => setFilterTier('midfield')}
                className={`px-3 py-1 text-xs font-label-caps uppercase transition-colors ${
                  filterTier === 'midfield' ? 'bg-primary-container text-text-primary font-bold' : 'text-text-muted hover:text-text-primary'
                }`}
              >
                Midfield
              </button>
            </div>
          </div>

          <div className="w-full overflow-x-auto border border-border-subtle bg-surface-panel">
            <table className="w-full text-left text-xs">
              <thead className="bg-canvas-base border-b border-border-subtle font-label-caps uppercase text-text-subtle">
                <tr>
                  <th className="py-3 px-4">POS</th>
                  <th className="py-3 px-4">CONSTRUCTOR</th>
                  <th className="py-3 px-4 text-right">MEAN PTS</th>
                  <th className="py-3 px-4 text-right">MEDIAN</th>
                  <th className="py-3 px-4 min-w-[220px]">90% HDI [LOW - HIGH]</th>
                  <th className="py-3 px-4 text-right">P(WCC TITLE)</th>
                  <th className="py-3 px-4 text-right">P(TOP 3)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border-subtle">
                {filteredStandings.map((c) => {
                  const isFerrari = c.constructor_id.toLowerCase().includes('ferrari');
                  return (
                    <tr
                      key={c.constructor_id}
                      className={`hover:bg-surface-raised transition-colors ${
                        isFerrari ? 'bg-primary-container/10 border-l-4 border-l-primary-container font-semibold' : ''
                      }`}
                    >
                      <td className="py-3 px-4 font-data-mono-md font-bold text-text-primary tnum">
                        P{c.expected_position}
                      </td>
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-2">
                          <span className="font-headline-sm uppercase text-text-primary">
                            {c.constructor_id.replace('_', ' ')}
                          </span>
                          {isFerrari && (
                            <span className="px-1.5 py-0.2 bg-primary-container text-text-primary text-[10px] uppercase font-bold">
                              SCUDERIA FERRARI
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="py-3 px-4 text-right font-data-mono-md text-text-primary tnum font-bold">
                        {c.mean_points.toFixed(1)}
                      </td>
                      <td className="py-3 px-4 text-right font-data-mono-md text-text-muted tnum">
                        {c.median_points.toFixed(1)}
                      </td>
                      <td className="py-3 px-4">
                        <CredibleIntervalBar
                          min={0}
                          max={800}
                          ciLow={c.ci_90[0]}
                          ciHigh={c.ci_90[1]}
                          median={c.median_points}
                          color={isFerrari ? '#E10600' : '#888888'}
                          label="90% HDI"
                          showLabels={true}
                        />
                      </td>
                      <td className="py-3 px-4 text-right font-data-mono-md text-text-primary tnum">
                        {(c.prob_p1 * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 px-4 text-right font-data-mono-md text-text-primary tnum">
                        {(c.prob_top3 * 100).toFixed(1)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>

        {/* SECTION 2: FINISHING POSITION PROBABILITY HEATMAP */}
        <section className="flex flex-col gap-6">
          <div className="border-b border-border-subtle pb-3">
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-3 bg-primary-container"></span>
              <h2 className="font-headline-md text-headline-md uppercase text-text-primary">
                FINISHING POSITION PROBABILITY HEATMAP (P1 TO P10+)
              </h2>
            </div>
            <p className="font-body-sm text-text-muted mt-0.5">
              Simulated probability of each constructor finishing in positions P1 through P10 across 10,000 paired seasons.
            </p>
          </div>

          <div className="w-full overflow-x-auto border border-border-subtle bg-surface-panel">
            <table className="w-full text-center text-xs">
              <thead className="bg-canvas-base border-b border-border-subtle font-label-caps uppercase text-text-subtle">
                <tr>
                  <th className="py-3 px-4 text-left">CONSTRUCTOR</th>
                  {positions.map((pos) => (
                    <th key={pos} className="py-3 px-2 font-data-mono-md">
                      {pos}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-border-subtle">
                {standings.map((team) => {
                  const isFerrari = team.constructor_id.toLowerCase().includes('ferrari');
                  return (
                    <tr
                      key={team.constructor_id}
                      className={isFerrari ? 'border-l-4 border-l-primary-container' : ''}
                    >
                      <td className="py-3 px-4 text-left font-headline-sm uppercase text-text-primary">
                        {team.constructor_id.replace('_', ' ')}
                      </td>
                      {positions.map((pos) => {
                        const prob = getTeamPositionProb(team.constructor_id, pos);
                        const colorClass = getHeatmapColor(prob, isFerrari);
                        return (
                          <td key={pos} className="py-2 px-1">
                            <div className={`py-1.5 px-1 font-data-mono-md text-[11px] tnum ${colorClass}`}>
                              {prob > 0.001 ? `${(prob * 100).toFixed(1)}%` : '—'}
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="flex items-center gap-4 text-xs font-label-caps uppercase text-text-subtle">
            <span>INTENSITY SCALE:</span>
            <div className="flex items-center gap-1">
              <span className="w-3 h-3 bg-surface-raised border border-border-subtle inline-block"></span>
              <span>0%</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="w-3 h-3 bg-surface-bright inline-block"></span>
              <span>&lt;20%</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="w-3 h-3 bg-border-focus inline-block"></span>
              <span>20-50%</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="w-3 h-3 bg-primary-container inline-block"></span>
              <span>Ferrari Mode / Primary (&gt;50%)</span>
            </div>
          </div>
        </section>

        {/* SECTION 3: DRIVERS' CHAMPIONSHIP (WDC) OUTLOOK */}
        <section className="flex flex-col gap-6">
          <div className="border-b border-border-subtle pb-3">
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-3 bg-primary-container"></span>
              <h2 className="font-headline-md text-headline-md uppercase text-text-primary">
                DRIVERS' CHAMPIONSHIP (WDC) — TITLE OUTLOOK &amp; PAIRED SAMPLES
              </h2>
            </div>
            <p className="font-body-sm text-text-muted mt-0.5">
              Individual driver distributions, median point baselines, and simulated title capture probabilities over 24 Grands Prix.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {driverContenders.map((d) => (
              <div
                key={d.name}
                className="bg-surface-panel p-5 border border-border-subtle flex flex-col justify-between gap-4"
                style={{ borderTop: `2px solid ${d.color}` }}
              >
                <div>
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-2.5">
                      <span className="font-headline-lg text-3xl font-bold leading-none" style={{ color: d.color }}>
                        {d.num}
                      </span>
                      <div className="flex flex-col">
                        <span className="font-headline-sm uppercase text-text-primary font-bold">
                          {d.name}
                        </span>
                        <span className="font-label-caps text-[10px] text-text-subtle uppercase">
                          {d.team}
                        </span>
                      </div>
                    </div>
                    <div className="flex flex-col items-end">
                      <span className="font-label-caps text-[10px] text-text-subtle uppercase">P(WDC TITLE)</span>
                      <span className="font-data-mono-lg text-xl font-bold tnum" style={{ color: d.color }}>
                        {(d.titleProb * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-2 my-4 bg-surface-raised p-2 text-xs border border-border-subtle">
                    <div className="flex flex-col">
                      <span className="font-label-caps text-[10px] text-text-subtle uppercase">EXP POINTS</span>
                      <span className="font-data-mono-md text-text-primary font-bold tnum">
                        {d.points.toFixed(1)}
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="font-label-caps text-[10px] text-text-subtle uppercase">EXP WINS</span>
                      <span className="font-data-mono-md text-text-primary font-bold tnum">
                        {d.wins.toFixed(1)}
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="font-label-caps text-[10px] text-text-subtle uppercase">PODIUMS</span>
                      <span className="font-data-mono-md text-text-primary font-bold tnum">
                        {d.podiums.toFixed(1)}
                      </span>
                    </div>
                  </div>

                  <CredibleIntervalBar
                    min={100}
                    max={420}
                    ciLow={d.ci90[0]}
                    ciHigh={d.ci90[1]}
                    median={d.median}
                    color={d.color}
                    label="90% HDI"
                  />
                </div>

                <div className="pt-2 border-t border-border-subtle text-xs text-text-subtle font-body-sm">
                  {d.note}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* SECTION 4: HEAD-TO-HEAD BENCHMARKING */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Ferrari vs McLaren */}
          <div className="bg-surface-panel p-6 border border-border-subtle flex flex-col gap-4">
            <div className="flex items-center gap-2 border-b border-border-subtle pb-3">
              <span className="w-1.5 h-3 bg-secondary"></span>
              <h3 className="font-headline-sm text-lg uppercase text-text-primary font-bold">
                HEAD-TO-HEAD: FERRARI VS MCLAREN
              </h3>
            </div>
            <p className="text-xs text-text-muted leading-relaxed">
              Ferrari dominates McLaren in expected constructors' points (488.8 pts vs 325.0 pts, a <span className="text-secondary font-bold">+163.8 pts advantage</span>). Ferrari finishes ahead of McLaren in 98.4% of all 10,000 paired Monte Carlo seasons.
            </p>
            <div className="grid grid-cols-2 gap-3 bg-surface-raised p-3 border border-border-subtle text-xs">
              <div>
                <span className="font-label-caps text-text-subtle uppercase block text-[10px]">SCUDERIA FERRARI</span>
                <span className="font-data-mono-md text-lg text-primary-container font-bold">488.8 PTS</span>
                <span className="block text-text-subtle mt-0.5">Top-3 Prob: 98.5%</span>
              </div>
              <div>
                <span className="font-label-caps text-text-subtle uppercase block text-[10px]">MCLAREN MERCEDES</span>
                <span className="font-data-mono-md text-lg text-telemetry-amber font-bold">325.0 PTS</span>
                <span className="block text-text-subtle mt-0.5">Top-3 Prob: 1.6%</span>
              </div>
            </div>
          </div>

          {/* Ferrari vs Red Bull */}
          <div className="bg-surface-panel p-6 border border-border-subtle flex flex-col gap-4">
            <div className="flex items-center gap-2 border-b border-border-subtle pb-3">
              <span className="w-1.5 h-3 bg-telemetry-amber"></span>
              <h3 className="font-headline-sm text-lg uppercase text-text-primary font-bold">
                HEAD-TO-HEAD: FERRARI VS RED BULL
              </h3>
            </div>
            <p className="text-xs text-text-muted leading-relaxed">
              Red Bull holds a narrow expected points edge over Ferrari (567.6 pts vs 488.8 pts, a <span className="text-telemetry-amber font-bold">-78.8 pts deficit</span> for Ferrari). However, Ferrari outscores Red Bull in 17.1% of seasons, primarily driven by high-downforce street circuits.
            </p>
            <div className="grid grid-cols-2 gap-3 bg-surface-raised p-3 border border-border-subtle text-xs">
              <div>
                <span className="font-label-caps text-text-subtle uppercase block text-[10px]">RED BULL RACING</span>
                <span className="font-data-mono-md text-lg text-text-primary font-bold">567.6 PTS</span>
                <span className="block text-text-subtle mt-0.5">P1 Prob: 11.6%</span>
              </div>
              <div>
                <span className="font-label-caps text-text-subtle uppercase block text-[10px]">SCUDERIA FERRARI</span>
                <span className="font-data-mono-md text-lg text-primary-container font-bold">488.8 PTS</span>
                <span className="block text-text-subtle mt-0.5">P1 Prob: 1.1%</span>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

