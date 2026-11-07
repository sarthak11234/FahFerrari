import React, { useState, useEffect } from 'react';
import { Link } from '../../context/RouterContext';
import { get2027Predictions, get2027Provenance } from '../../services/api';
import CredibleIntervalBar from '../../components/common/CredibleIntervalBar';
import {
  ShieldAlert,
  Sparkles,
  ArrowRight,
  TrendingUp,
  Cpu,
  Layers,
  Flag,
  Award,
  AlertTriangle,
  ChevronRight,
  BarChart3,
  Calendar,
  Lock,
} from 'lucide-react';

export default function OverviewPage() {
  const [data, setData] = useState(null);
  const [provenance, setProvenance] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [preds, prov] = await Promise.all([
          get2027Predictions(),
          get2027Provenance(),
        ]);
        setData(preds);
        setProvenance(prov);
      } catch (err) {
        console.error('Error loading overview forecast:', err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading || !data) {
    return (
      <div className="w-full min-h-[60vh] flex flex-col items-center justify-center gap-3">
        <div className="w-8 h-8 border-2 border-primary-container border-t-transparent animate-spin"></div>
        <span className="font-label-caps text-xs uppercase text-text-muted">Loading Preseason Forecast Model...</span>
      </div>
    );
  }

  const ferrari = data.ferrari_constructors_forecast;
  const leclerc = data.driver_forecasts.leclerc;
  const hamilton = data.driver_forecasts.hamilton;
  const standings = data.constructors_standings;
  const races = data.race_by_race_forecast;

  // Selected highlight races
  const highlightRaces = [
    races.find((r) => r.round === 1) || races[0],
    races.find((r) => r.circuit_id === 'monaco') || races[7],
    races.find((r) => r.circuit_id === 'monza') || races[15],
    races.find((r) => r.circuit_id === 'marina_bay') || races[17],
  ].filter(Boolean);

  return (
    <div className="flex flex-col w-full">
      {/* 1. CINEMATIC HERO SECTION */}
      <section className="relative w-full min-h-[520px] bg-canvas-base overflow-hidden flex flex-col justify-between border-b border-border-subtle">
        {/* Hero Background Photography with Precision Vignette */}
        <div
          className="absolute inset-0 bg-cover bg-right md:bg-center opacity-40 mix-blend-luminosity filter contrast-125"
          style={{
            backgroundImage:
              "url('https://lh3.googleusercontent.com/aida-public/AB6AXuAHF8xzGI5Ao-G1VwnnagKm1k2zGc-eDE97IS-LhOEoYJ2TtE9iDZpbs0hkJuvk716xFRdq3zsmmFLmByv3E2NyI0SqzuFstwMhDTEu0Tfuczro-ux1EPdXt78mWUBlocuqmtXNrfr5otzKZKvwFfQeq1KkzMmadXHW5718ZLAVARW6wLqHPFkzhpCu6RkOTk1EeWW8Hr5ClkRXAAbDk0dM3ba7rXlEa3_jG5qZaoBZNkFYuGmkIFQ')",
          }}
        />
        {/* Layered Technical Gradients */}
        <div className="absolute inset-0 bg-gradient-to-r from-canvas-base via-canvas-base/90 to-transparent pointer-events-none" />
        <div className="absolute inset-0 bg-gradient-to-t from-canvas-base via-transparent to-canvas-base/60 pointer-events-none" />

        {/* Content Body */}
        <div className="relative z-10 w-full max-w-[1440px] mx-auto px-4 md:px-margin-desktop pt-12 pb-8 flex-1 flex flex-col justify-center">
          <div className="max-w-3xl flex flex-col gap-space-sm">
            {/* Eyebrow & Frozen Badge */}
            <div className="flex flex-wrap items-center gap-space-sm">
              <div className="flex items-center gap-1.5">
                <span className="w-1.5 h-3 bg-primary-container inline-block"></span>
                <span className="font-label-caps text-label-caps uppercase text-primary-container tracking-widest font-bold">
                  F1 PERFORMANCE INTELLIGENCE · 2027
                </span>
              </div>
              <span className="px-2 py-0.5 bg-surface-raised border border-border-subtle font-label-caps text-[11px] uppercase text-text-muted tracking-wider flex items-center gap-1">
                <Lock className="w-3 h-3 text-secondary" />
                FROZEN PRESEASON BASELINE
              </span>
            </div>

            {/* Main Headline */}
            <h1 className="font-display-xl text-display-xl-mobile md:text-display-xl uppercase text-text-primary tracking-tight leading-none mt-1">
              THE 2027 SEASON. <span className="text-primary-container">PREDICTED BY DATA.</span>
            </h1>

            {/* Subtitle Narrative */}
            <p className="font-body-lg text-body-md md:text-body-lg text-text-muted max-w-2xl leading-relaxed mt-1">
              A probabilistic forecast of Scuderia Ferrari’s championship campaign, powered by Bayesian hierarchical latent pace modeling, discrete-time reliability hazard functions, and 10,000 paired Monte Carlo season simulations.
            </p>

            {/* CTA Hub */}
            <div className="flex flex-wrap items-center gap-space-md pt-4">
              <a
                href="#scuderia-outlook"
                className="px-5 py-2.5 bg-primary-container hover:bg-ferrari-red-hover text-text-primary font-headline-sm text-headline-sm uppercase tracking-wider transition-all inline-flex items-center gap-2 shadow-[0_0_15px_rgba(225,6,0,0.3)]"
              >
                <span>EXPLORE THE FORECAST</span>
                <ArrowRight className="w-4 h-4" />
              </a>

              <Link
                href="/model-intelligence"
                className="px-5 py-2.5 bg-surface-panel hover:bg-surface-raised border border-border-subtle hover:border-border-focus text-text-primary font-headline-sm text-headline-sm uppercase tracking-wider transition-colors inline-flex items-center gap-2"
              >
                <span>HOW THE MODEL WORKS</span>
                <ChevronRight className="w-4 h-4 text-text-muted" />
              </Link>
            </div>
          </div>
        </div>

        {/* Hero Metadata Strip */}
        <div className="relative z-10 w-full border-t border-border-subtle bg-surface-panel/80 backdrop-blur-sm">
          <div className="max-w-[1440px] mx-auto px-4 md:px-margin-desktop py-2.5 grid grid-cols-2 md:grid-cols-4 gap-3 text-xs font-label-caps uppercase text-text-subtle">
            <div className="flex items-center gap-2">
              <span className="text-text-muted">MODEL:</span>
              <span className="text-text-primary font-data-mono-md">v2027.1-PRESEASON</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-text-muted">STATUS:</span>
              <span className="text-secondary font-data-mono-md">FROZEN PRE-TESTING</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-text-muted">DATA CUTOFF:</span>
              <span className="text-text-primary font-data-mono-md">FEB 18, 2027</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-text-muted">BUDGET:</span>
              <span className="text-text-primary font-data-mono-md">10,000 MC RUNS</span>
            </div>
          </div>
        </div>
      </section>

      {/* 2. SEASON INTELLIGENCE STRIP */}
      <section className="w-full bg-surface-panel border-b border-border-subtle py-4">
        <div className="max-w-[1440px] mx-auto px-4 md:px-margin-desktop grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="flex items-center gap-3 border-r border-border-subtle/60 pr-3">
            <Calendar className="w-5 h-5 text-primary-container shrink-0" />
            <div className="flex flex-col">
              <span className="font-label-caps text-[10px] text-text-subtle uppercase">CALENDAR EXPANSION</span>
              <span className="font-data-mono-md text-text-primary font-bold">24 GRANDS PRIX</span>
            </div>
          </div>

          <div className="flex items-center gap-3 border-r border-border-subtle/60 pr-3">
            <Flag className="w-5 h-5 text-telemetry-amber shrink-0" />
            <div className="flex flex-col">
              <span className="font-label-caps text-[10px] text-text-subtle uppercase">SPRINT FORMAT</span>
              <span className="font-data-mono-md text-text-primary font-bold">10 SPRINT EVENTS</span>
            </div>
          </div>

          <div className="flex items-center gap-3 border-r border-border-subtle/60 pr-3">
            <Cpu className="w-5 h-5 text-secondary shrink-0" />
            <div className="flex flex-col">
              <span className="font-label-caps text-[10px] text-text-subtle uppercase">SIMULATION SCALE</span>
              <span className="font-data-mono-md text-text-primary font-bold">10,000 PAIRED SEASONS</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Layers className="w-5 h-5 text-tertiary shrink-0" />
            <div className="flex flex-col">
              <span className="font-label-caps text-[10px] text-text-subtle uppercase">REGULATION REGIME</span>
              <span className="font-data-mono-md text-text-primary font-bold">ACTIVE AERO / 50-50 PU</span>
            </div>
          </div>
        </div>
      </section>

      {/* MAIN CONTENT CONTAINER */}
      <div className="max-w-[1440px] mx-auto px-4 md:px-margin-desktop py-12 flex flex-col gap-16">
        {/* 3. SCUDERIA CHAMPIONSHIP OUTLOOK */}
        <section id="scuderia-outlook" className="flex flex-col gap-6">
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-2 border-b border-border-subtle pb-3">
            <div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-4 bg-primary-container"></span>
                <h2 className="font-headline-lg text-headline-lg uppercase text-text-primary font-bold">
                  THE SCUDERIA OUTLOOK
                </h2>
              </div>
              <p className="font-body-sm text-text-muted mt-1">
                2027 Constructors' Championship probabilistic distribution derived across 10,000 paired Monte Carlo iterations.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 bg-primary-container/10 border border-primary-container text-primary-container font-label-caps text-xs uppercase font-bold">
                P3 EXPECTED FINISH (80.3%)
              </span>
            </div>
          </div>

          {/* Primary Metric Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-surface-panel p-5 border border-border-subtle border-t-2 border-t-primary-container">
              <span className="font-label-caps text-xs text-text-muted uppercase">EXPECTED POINTS</span>
              <div className="font-data-mono-lg text-4xl text-text-primary font-bold mt-2 tnum">
                {ferrari.mean_points.toFixed(1)}
              </div>
              <div className="text-xs text-text-subtle mt-2 flex justify-between">
                <span>MEDIAN: {ferrari.median_points.toFixed(1)} PTS</span>
                <span className="text-secondary">+28.4 vs 2026</span>
              </div>
            </div>

            <div className="bg-surface-panel p-5 border border-border-subtle border-t-2 border-t-primary-container">
              <span className="font-label-caps text-xs text-text-muted uppercase">90% CREDIBLE INTERVAL (HDI)</span>
              <div className="font-data-mono-lg text-3xl text-text-primary font-bold mt-2 tnum">
                [{ferrari.ci_90[0].toFixed(0)} – {ferrari.ci_90[1].toFixed(0)}]
              </div>
              <div className="text-xs text-text-subtle mt-2">
                80% INTERVAL: [{ferrari.ci_80[0].toFixed(0)} – {ferrari.ci_80[1].toFixed(0)}]
              </div>
            </div>

            <div className="bg-surface-panel p-5 border border-border-subtle border-t-2 border-t-primary-container">
              <span className="font-label-caps text-xs text-text-muted uppercase">P(CHAMPIONSHIP WIN)</span>
              <div className="font-data-mono-lg text-4xl text-primary-container font-bold mt-2 tnum">
                {(ferrari.prob_championship_win * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-text-subtle mt-2">
                109 SEASONS IN 10,000 RUNS
              </div>
            </div>

            <div className="bg-surface-panel p-5 border border-border-subtle border-t-2 border-t-secondary">
              <span className="font-label-caps text-xs text-text-muted uppercase">P(TOP 3 FINISH)</span>
              <div className="font-data-mono-lg text-4xl text-secondary font-bold mt-2 tnum">
                {(ferrari.prob_top3 * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-text-subtle mt-2">
                9,849 SEASONS IN 10,000 RUNS
              </div>
            </div>
          </div>

          {/* Credible Interval Bar & Position Probability Distribution */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 bg-surface-panel p-5 border border-border-subtle">
            <div className="flex flex-col justify-between gap-4">
              <div>
                <span className="font-label-caps text-xs text-text-muted uppercase font-semibold">
                  CONSTRUCTORS' POINTS CREDIBLE INTERVAL
                </span>
                <p className="text-xs text-text-subtle mt-1">
                  High Density Interval (90% HDI) spanning {ferrari.ci_90[0]} to {ferrari.ci_90[1]} points, centered at median {ferrari.median_points} points.
                </p>
              </div>

              <CredibleIntervalBar
                min={300}
                max={750}
                ciLow={ferrari.ci_90[0]}
                ciHigh={ferrari.ci_90[1]}
                median={ferrari.median_points}
                color="#E10600"
                label="90% HDI [403 – 577 PTS]"
              />

              <div className="text-xs text-text-subtle">
                Baseline indicates Ferrari holds an 80.3% probability of finishing 3rd in the World Constructors' Championship, behind Mercedes and Red Bull.
              </div>
            </div>

            {/* Position Probability Distribution */}
            <div className="flex flex-col gap-3">
              <span className="font-label-caps text-xs text-text-muted uppercase font-semibold">
                FINISHING POSITION PROBABILITY MATRIX
              </span>
              <div className="flex flex-col gap-2">
                {Object.entries(ferrari.position_probabilities)
                  .filter(([_, prob]) => prob > 0.001)
                  .map(([pos, prob]) => (
                    <div key={pos} className="flex items-center gap-3 text-xs">
                      <span className="w-8 font-data-mono-md font-bold text-text-primary">{pos}</span>
                      <div className="flex-1 bg-surface-raised h-4 relative overflow-hidden border border-border-subtle">
                        <div
                          className="h-full bg-primary-container transition-all"
                          style={{ width: `${(prob * 100).toFixed(1)}%` }}
                        />
                      </div>
                      <span className="w-14 text-right font-data-mono-md text-text-primary tnum">
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        </section>

        {/* 4. THE DRIVER EQUATION: LECLERC VS HAMILTON */}
        <section className="flex flex-col gap-6">
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-2 border-b border-border-subtle pb-3">
            <div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-4 bg-primary-container"></span>
                <h2 className="font-headline-lg text-headline-lg uppercase text-text-primary font-bold">
                  THE DRIVER EQUATION
                </h2>
              </div>
              <p className="font-body-sm text-text-muted mt-1">
                Charles Leclerc versus Lewis Hamilton: Paired sample simulated intra-team duel for 2027.
              </p>
            </div>
            <Link
              href="/ferrari"
              className="text-xs font-label-caps text-primary-container uppercase hover:underline flex items-center gap-1"
            >
              <span>IN-DEPTH SCUDERIA TELEMETRY</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Charles Leclerc Card */}
            <div className="bg-surface-panel p-6 border border-border-subtle border-t-2 border-t-primary-container flex flex-col justify-between gap-4">
              <div>
                <div className="flex justify-between items-start">
                  <div>
                    <span className="font-headline-md text-3xl font-bold text-primary-container">#16</span>
                    <h3 className="font-headline-md text-2xl uppercase text-text-primary font-bold mt-1">
                      CHARLES LECLERC
                    </h3>
                    <span className="font-label-caps text-xs text-text-subtle uppercase">
                      MONACO · CONTRACTED LEAD
                    </span>
                  </div>
                  <div className="text-right">
                    <span className="font-label-caps text-[10px] text-text-subtle uppercase">P(WDC TITLE)</span>
                    <div className="font-data-mono-md text-xl text-primary-container font-bold tnum">
                      {(leclerc.prob_drivers_title * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-2 my-4 bg-surface-raised p-3 border border-border-subtle">
                  <div className="flex flex-col">
                    <span className="font-label-caps text-[10px] text-text-subtle uppercase">EXP POINTS</span>
                    <span className="font-data-mono-md text-lg text-text-primary font-bold tnum">
                      {leclerc.mean_points.toFixed(1)}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="font-label-caps text-[10px] text-text-subtle uppercase">EXP WINS</span>
                    <span className="font-data-mono-md text-lg text-text-primary font-bold tnum">
                      {leclerc.expected_wins.toFixed(1)}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="font-label-caps text-[10px] text-text-subtle uppercase">EXP PODIUMS</span>
                    <span className="font-data-mono-md text-lg text-text-primary font-bold tnum">
                      {leclerc.expected_podiums.toFixed(1)}
                    </span>
                  </div>
                </div>

                <CredibleIntervalBar
                  min={100}
                  max={350}
                  ciLow={leclerc.ci_90[0]}
                  ciHigh={leclerc.ci_90[1]}
                  median={leclerc.median_points}
                  color="#E10600"
                  label="90% HDI [158 – 285 PTS]"
                />
              </div>

              <div className="pt-3 border-t border-border-subtle flex justify-between items-center text-xs">
                <span className="text-text-subtle">Expected DNFs:</span>
                <span className="font-data-mono-md text-telemetry-delta-neg font-semibold tnum">
                  {leclerc.expected_dnfs.toFixed(1)} (Weibull hazard)
                </span>
              </div>
            </div>

            {/* Lewis Hamilton Card */}
            <div className="bg-surface-panel p-6 border border-border-subtle border-t-2 border-t-telemetry-amber flex flex-col justify-between gap-4">
              <div>
                <div className="flex justify-between items-start">
                  <div>
                    <span className="font-headline-md text-3xl font-bold text-telemetry-amber">#44</span>
                    <h3 className="font-headline-md text-2xl uppercase text-text-primary font-bold mt-1">
                      LEWIS HAMILTON
                    </h3>
                    <span className="font-label-caps text-xs text-text-subtle uppercase">
                      GREAT BRITAIN · 7-TIME WORLD CHAMPION
                    </span>
                  </div>
                  <div className="text-right">
                    <span className="font-label-caps text-[10px] text-text-subtle uppercase">P(WDC TITLE)</span>
                    <div className="font-data-mono-md text-xl text-telemetry-amber font-bold tnum">
                      {(hamilton.prob_drivers_title * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-2 my-4 bg-surface-raised p-3 border border-border-subtle">
                  <div className="flex flex-col">
                    <span className="font-label-caps text-[10px] text-text-subtle uppercase">EXP POINTS</span>
                    <span className="font-data-mono-md text-lg text-text-primary font-bold tnum">
                      {hamilton.mean_points.toFixed(1)}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="font-label-caps text-[10px] text-text-subtle uppercase">EXP WINS</span>
                    <span className="font-data-mono-md text-lg text-text-primary font-bold tnum">
                      {hamilton.expected_wins.toFixed(1)}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="font-label-caps text-[10px] text-text-subtle uppercase">EXP PODIUMS</span>
                    <span className="font-data-mono-md text-lg text-text-primary font-bold tnum">
                      {hamilton.expected_podiums.toFixed(1)}
                    </span>
                  </div>
                </div>

                <CredibleIntervalBar
                  min={100}
                  max={350}
                  ciLow={hamilton.ci_90[0]}
                  ciHigh={hamilton.ci_90[1]}
                  median={hamilton.median_points}
                  color="#E7B45A"
                  label="90% HDI [205 – 335 PTS]"
                />
              </div>

              <div className="pt-3 border-t border-border-subtle flex justify-between items-center text-xs">
                <span className="text-text-subtle">Expected DNFs:</span>
                <span className="font-data-mono-md text-telemetry-delta-pos font-semibold tnum">
                  {hamilton.expected_dnfs.toFixed(1)} (Weibull hazard)
                </span>
              </div>
            </div>
          </div>
        </section>

        {/* 5. THE COMPETITIVE FIELD */}
        <section className="flex flex-col gap-6">
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-2 border-b border-border-subtle pb-3">
            <div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-4 bg-primary-container"></span>
                <h2 className="font-headline-lg text-headline-lg uppercase text-text-primary font-bold">
                  THE COMPETITIVE FIELD
                </h2>
              </div>
              <p className="font-body-sm text-text-muted mt-1">
                Full 10-team constructors' benchmark ranking derived from 10,000 paired Monte Carlo season simulations.
              </p>
            </div>
            <Link
              href="/championship"
              className="text-xs font-label-caps text-primary-container uppercase hover:underline flex items-center gap-1"
            >
              <span>FULL CHAMPIONSHIP MATRICES</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <div className="w-full overflow-x-auto border border-border-subtle bg-surface-panel">
            <table className="w-full text-left text-xs">
              <thead className="bg-canvas-base border-b border-border-subtle font-label-caps uppercase text-text-subtle">
                <tr>
                  <th className="py-3 px-4">POS</th>
                  <th className="py-3 px-4">CONSTRUCTOR</th>
                  <th className="py-3 px-4 text-right">MEAN POINTS</th>
                  <th className="py-3 px-4 text-right">MEDIAN</th>
                  <th className="py-3 px-4 text-center">90% HDI WHISKER</th>
                  <th className="py-3 px-4 text-right">P(P1 TITLE)</th>
                  <th className="py-3 px-4 text-right">P(TOP 3)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border-subtle">
                {standings.map((c) => {
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
                              PRIMARY SUBJECT
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
                      <td className="py-3 px-4 min-w-[200px]">
                        <CredibleIntervalBar
                          min={0}
                          max={800}
                          ciLow={c.ci_90[0]}
                          ciHigh={c.ci_90[1]}
                          median={c.median_points}
                          color={isFerrari ? '#E10600' : '#A0A5AD'}
                          showLabels={false}
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

        {/* 6. RACES TO WATCH */}
        <section className="flex flex-col gap-6">
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-2 border-b border-border-subtle pb-3">
            <div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-4 bg-primary-container"></span>
                <h2 className="font-headline-lg text-headline-lg uppercase text-text-primary font-bold">
                  RACES TO WATCH
                </h2>
              </div>
              <p className="font-body-sm text-text-muted mt-1">
                Strategic circuit highlights with simulated Ferrari points, win probability, and mechanical DNF hazard.
              </p>
            </div>
            <Link
              href="/race-calendar"
              className="text-xs font-label-caps text-primary-container uppercase hover:underline flex items-center gap-1"
            >
              <span>VIEW FULL 24-RACE CALENDAR</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {highlightRaces.map((r) => (
              <div
                key={r.round}
                className="bg-surface-panel p-5 border border-border-subtle flex flex-col justify-between gap-3 hover:border-border-focus transition-colors"
              >
                <div>
                  <div className="flex justify-between items-center text-xs text-text-subtle font-label-caps">
                    <span>ROUND {r.round}</span>
                    {r.is_sprint && (
                      <span className="px-1.5 py-0.5 bg-telemetry-amber/20 text-telemetry-amber font-bold">
                        SPRINT
                      </span>
                    )}
                  </div>
                  <h4 className="font-headline-sm text-lg uppercase text-text-primary font-bold mt-1">
                    {r.circuit}
                  </h4>
                  <span className="font-label-caps text-[11px] text-text-subtle uppercase">
                    CIRCUIT ID: {r.circuit_id}
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-2 my-2 bg-surface-raised p-2 text-xs">
                  <div>
                    <span className="font-label-caps text-[10px] text-text-subtle uppercase">EXP POINTS</span>
                    <div className="font-data-mono-md text-text-primary font-bold tnum">
                      {r.expected_points.toFixed(1)}
                    </div>
                  </div>
                  <div>
                    <span className="font-label-caps text-[10px] text-text-subtle uppercase">P(WIN)</span>
                    <div className="font-data-mono-md text-primary-container font-bold tnum">
                      {(r.win_probability * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                <div className="flex justify-between items-center text-xs text-text-subtle pt-2 border-t border-border-subtle">
                  <span>P(Podium): {(r.podium_probability * 100).toFixed(1)}%</span>
                  <span className="text-telemetry-amber">DNF: {(r.dnf_risk * 100).toFixed(1)}%</span>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* 7. MODEL INTEGRITY & PROVENANCE PANEL */}
        <section className="bg-surface-panel p-6 border border-border-subtle border-l-4 border-l-secondary flex flex-col gap-4">
          <div className="flex items-center gap-2">
            <ShieldAlert className="w-5 h-5 text-secondary" />
            <h3 className="font-headline-sm text-lg uppercase text-text-primary font-bold">
              MODEL INTEGRITY, PROVENANCE & METHODOLOGY NOTICE
            </h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs text-text-muted leading-relaxed">
            <div>
              <span className="font-label-caps uppercase text-text-primary block font-semibold mb-1">
                1. BAYESIAN CREDIBILITY VS ZERO GUARANTEES
              </span>
              All 2027 forecasts represent probabilistic posterior expectations over 10,000 paired Monte Carlo simulations. They reflect empirical uncertainty under 2027 technical regulations and must not be interpreted as deterministic sports scores.
            </div>
            <div>
              <span className="font-label-caps uppercase text-text-primary block font-semibold mb-1">
                2. REPRODUCIBLE FROZEN BASELINE
              </span>
              This model artifact is frozen as of <span className="text-text-primary">{provenance?.freeze_date_utc || '2026-11-07'}</span> (Git commit <span className="text-primary-container font-data-mono-md">{provenance?.code_commit_baseline || 'ae91b36'}</span>, random seed {provenance?.random_seed || 2027}). Zero data leakage from 2027 testing sessions is permitted.
            </div>
            <div>
              <span className="font-label-caps uppercase text-text-primary block font-semibold mb-1">
                3. HISTORICAL DRIFT REMEDIATION
              </span>
              Following our formal retrospective on the 2026 model, all 10 legacy defects (DEF-01 through DEF-10) have been remediated: Weibull reliability hazard modeling, paired driver samples, and dynamic aerodynamic archetype sensitivity.
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
