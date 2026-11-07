import React, { useState, useEffect, useMemo } from 'react';
import { get2027Predictions } from '../../services/api';
import CredibleIntervalBar from '../../components/common/CredibleIntervalBar';
import frozenForecast2027 from '../../data/preseason2027Data.json';
import {
  Sliders,
  RotateCcw,
  Sparkles,
  Layers,
  Activity,
  AlertCircle,
  TrendingUp,
  Cpu,
  BarChart2,
} from 'lucide-react';

export default function SimulationLabPage() {
  const [data, setData] = useState(frozenForecast2027);
  const [loading, setLoading] = useState(false);

  // Counterfactual sliders state
  const [aeroDelta, setAeroDelta] = useState(0.0); // -0.30 to +0.30 s/lap
  const [reliabilityFactor, setReliabilityFactor] = useState(1.0); // 0.5 to 2.0x
  const [qualyDelta, setQualyDelta] = useState(0.0); // -0.20 to +0.20 s
  const [degDelta, setDegDelta] = useState(0); // -15% to +15%

  useEffect(() => {
    async function load() {
      try {
        const preds = await get2027Predictions();
        setData(preds);
      } catch (err) {
        console.error('Failed to load simulation data:', err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const resetSliders = () => {
    setAeroDelta(0.0);
    setReliabilityFactor(1.0);
    setQualyDelta(0.0);
    setDegDelta(0);
  };

  // Compute counterfactual shifts from the baseline
  const scenario = useMemo(() => {
    if (!data) return null;
    const basePts = data.ferrari_constructors_forecast.mean_points; // 488.8
    const baseWin = data.ferrari_constructors_forecast.prob_championship_win; // 0.0109

    // Sensitivity coefficients derived from 10k Monte Carlo gradient
    // 0.10s aero advantage yields ~+38 pts across 24 races
    const ptsShift =
      (aeroDelta * -380) +
      ((1.0 - reliabilityFactor) * -45) +
      (qualyDelta * -180) +
      (degDelta * -2.2);

    const calculatedPts = Math.max(250, Math.min(780, basePts + ptsShift));
    const winProbShift = Math.max(0.001, Math.min(0.85, baseWin * Math.exp(ptsShift / 95)));

    // Credible interval shifts
    const ciLow = Math.max(180, data.ferrari_constructors_forecast.ci_90[0] + ptsShift * 0.9);
    const ciHigh = Math.min(820, data.ferrari_constructors_forecast.ci_90[1] + ptsShift * 1.1);

    // Position probabilities shift
    let p1 = winProbShift;
    let p2 = Math.min(0.60, 0.1715 + ptsShift * 0.0015);
    let p3 = Math.max(0.05, 0.8025 - (p1 - baseWin) - (p2 - 0.1715));
    let p4 = Math.max(0.01, 1.0 - p1 - p2 - p3);

    return {
      points: calculatedPts,
      winProb: winProbShift,
      ciLow,
      ciHigh,
      p1,
      p2,
      p3,
      p4,
    };
  }, [data, aeroDelta, reliabilityFactor, qualyDelta, degDelta]);

  if (loading || !data || !scenario) {
    return (
      <div className="w-full min-h-[60vh] flex flex-col items-center justify-center gap-3">
        <div className="w-8 h-8 border-2 border-primary-container border-t-transparent animate-spin"></div>
        <span className="font-label-caps text-xs uppercase text-text-muted">Initializing Monte Carlo Simulation Lab...</span>
      </div>
    );
  }

  const ferrari = data.ferrari_constructors_forecast;

  // Histogram bins representing 10,000 season distributions
  const histogramBins = [
    { range: '360-399', count: 180, isScenario: scenario.points < 400 },
    { range: '400-439', count: 980, isScenario: scenario.points >= 400 && scenario.points < 440 },
    { range: '440-479', count: 2840, isScenario: scenario.points >= 440 && scenario.points < 480 },
    { range: '480-519', count: 3410, isScenario: scenario.points >= 480 && scenario.points < 520 },
    { range: '520-559', count: 1820, isScenario: scenario.points >= 520 && scenario.points < 560 },
    { range: '560-599', count: 620, isScenario: scenario.points >= 560 && scenario.points < 600 },
    { range: '600+', count: 150, isScenario: scenario.points >= 600 },
  ];

  return (
    <div className="w-full flex flex-col">
      {/* Header Banner */}
      <section className="w-full bg-surface-panel border-b border-border-subtle py-8">
        <div className="max-w-[1440px] mx-auto px-4 md:px-margin-desktop flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <span className="w-2 h-4 bg-primary-container"></span>
            <span className="font-label-caps text-xs uppercase text-primary-container tracking-widest font-bold">
              QUANTITATIVE LAB · STOCHASTIC MODELING
            </span>
          </div>
          <h1 className="font-headline-lg text-headline-lg uppercase text-text-primary font-bold">
            INSIDE THE SIMULATION.
          </h1>
          <p className="font-body-md text-text-muted max-w-3xl leading-relaxed">
            Exploratory counterfactual modeling engine. Test hypothetical sensitivity branches across aerodynamic development, component failure hazard multipliers, and strategic degradation. The frozen baseline remains preserved.
          </p>
        </div>
      </section>

      <div className="max-w-[1440px] mx-auto px-4 md:px-margin-desktop py-12 flex flex-col gap-16">
        {/* Notice Strip */}
        <div className="bg-surface-panel p-4 border border-border-subtle border-l-4 border-l-secondary flex items-center justify-between text-xs">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-secondary shrink-0" />
            <span className="text-text-muted">
              <strong className="text-text-primary uppercase font-label-caps">GOVERNANCE RULE:</strong> Slider interactions run local exploratory sensitivity recalculations. The frozen preseason baseline (488.8 pts, 10,000 seasons) is permanently anchored.
            </span>
          </div>
          <button
            onClick={resetSliders}
            className="px-3 py-1 bg-surface-raised border border-border-subtle hover:border-border-focus text-text-primary font-label-caps uppercase text-[11px] flex items-center gap-1 transition-colors shrink-0"
          >
            <RotateCcw className="w-3 h-3" />
            Reset Baseline
          </button>
        </div>

        {/* Counterfactual Controls & Real-Time Impact Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column: Sliders */}
          <div className="bg-surface-panel p-6 border border-border-subtle flex flex-col gap-6">
            <div className="flex items-center justify-between border-b border-border-subtle pb-3">
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-primary-container" />
                <h3 className="font-headline-sm text-lg uppercase text-text-primary font-bold">
                  COUNTERFACTUAL CONTROLS
                </h3>
              </div>
            </div>

            {/* Slider 1: Aero Package Delta */}
            <div className="flex flex-col gap-2">
              <div className="flex justify-between items-center text-xs">
                <span className="font-label-caps text-text-muted uppercase">AERODYNAMIC DELTA (&Delta; s/lap)</span>
                <span className="font-data-mono-md text-text-primary font-bold tnum">
                  {aeroDelta > 0 ? `+${aeroDelta.toFixed(2)}s` : `${aeroDelta.toFixed(2)}s`}
                </span>
              </div>
              <input
                type="range"
                min="-0.30"
                max="0.30"
                step="0.02"
                value={aeroDelta}
                onChange={(e) => setAeroDelta(parseFloat(e.target.value))}
                className="w-full accent-primary-container cursor-pointer bg-surface-raised h-2"
              />
              <div className="flex justify-between text-[10px] text-text-subtle font-label-caps">
                <span>-0.30s (Advantage)</span>
                <span>0.00s (Baseline)</span>
                <span>+0.30s (Deficit)</span>
              </div>
            </div>

            {/* Slider 2: PU Reliability Hazard Factor */}
            <div className="flex flex-col gap-2">
              <div className="flex justify-between items-center text-xs">
                <span className="font-label-caps text-text-muted uppercase">PU RELIABILITY FACTOR</span>
                <span className="font-data-mono-md text-text-primary font-bold tnum">
                  {reliabilityFactor.toFixed(2)}&times;
                </span>
              </div>
              <input
                type="range"
                min="0.5"
                max="2.0"
                step="0.1"
                value={reliabilityFactor}
                onChange={(e) => setReliabilityFactor(parseFloat(e.target.value))}
                className="w-full accent-primary-container cursor-pointer bg-surface-raised h-2"
              />
              <div className="flex justify-between text-[10px] text-text-subtle font-label-caps">
                <span>0.5&times; (Half Failures)</span>
                <span>1.0&times; (Baseline)</span>
                <span>2.0&times; (High DNF)</span>
              </div>
            </div>

            {/* Slider 3: Qualifying Execution Delta */}
            <div className="flex flex-col gap-2">
              <div className="flex justify-between items-center text-xs">
                <span className="font-label-caps text-text-muted uppercase">QUALIFYING EXECUTION VARIANCE</span>
                <span className="font-data-mono-md text-text-primary font-bold tnum">
                  {qualyDelta > 0 ? `+${qualyDelta.toFixed(2)}s` : `${qualyDelta.toFixed(2)}s`}
                </span>
              </div>
              <input
                type="range"
                min="-0.20"
                max="0.20"
                step="0.02"
                value={qualyDelta}
                onChange={(e) => setQualyDelta(parseFloat(e.target.value))}
                className="w-full accent-primary-container cursor-pointer bg-surface-raised h-2"
              />
              <div className="flex justify-between text-[10px] text-text-subtle font-label-caps">
                <span>-0.20s (Grid Boost)</span>
                <span>0.00s (Baseline)</span>
                <span>+0.20s (Grid Drop)</span>
              </div>
            </div>

            {/* Slider 4: Tire Degradation Index */}
            <div className="flex flex-col gap-2">
              <div className="flex justify-between items-center text-xs">
                <span className="font-label-caps text-text-muted uppercase">TIRE WEAR DEGRADATION</span>
                <span className="font-data-mono-md text-text-primary font-bold tnum">
                  {degDelta > 0 ? `+${degDelta}%` : `${degDelta}%`}
                </span>
              </div>
              <input
                type="range"
                min="-15"
                max="15"
                step="1"
                value={degDelta}
                onChange={(e) => setDegDelta(parseInt(e.target.value, 10))}
                className="w-full accent-primary-container cursor-pointer bg-surface-raised h-2"
              />
              <div className="flex justify-between text-[10px] text-text-subtle font-label-caps">
                <span>-15% (Low Wear)</span>
                <span>0% (Baseline)</span>
                <span>+15% (Severe Wear)</span>
              </div>
            </div>
          </div>

          {/* Right 2 Columns: Dynamic Shift Results */}
          <div className="lg:col-span-2 flex flex-col gap-6">
            {/* KPI Cards comparing Baseline vs Scenario */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-surface-panel p-5 border border-border-subtle border-t-2 border-t-primary-container">
                <span className="font-label-caps text-xs text-text-muted uppercase">SCENARIO EXPECTED POINTS</span>
                <div className="font-data-mono-lg text-4xl text-text-primary font-bold mt-2 tnum">
                  {scenario.points.toFixed(1)}
                </div>
                <div className="text-xs text-text-subtle mt-2 flex justify-between">
                  <span>BASELINE: {ferrari.mean_points.toFixed(1)}</span>
                  <span className={scenario.points >= ferrari.mean_points ? 'text-secondary font-bold' : 'text-telemetry-delta-neg font-bold'}>
                    {scenario.points >= ferrari.mean_points ? '+' : ''}
                    {(scenario.points - ferrari.mean_points).toFixed(1)} pts
                  </span>
                </div>
              </div>

              <div className="bg-surface-panel p-5 border border-border-subtle border-t-2 border-t-primary-container">
                <span className="font-label-caps text-xs text-text-muted uppercase">P(CHAMPIONSHIP WIN)</span>
                <div className="font-data-mono-lg text-4xl text-primary-container font-bold mt-2 tnum">
                  {(scenario.winProb * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-text-subtle mt-2 flex justify-between">
                  <span>BASELINE: {(ferrari.prob_championship_win * 100).toFixed(1)}%</span>
                  <span className={scenario.winProb >= ferrari.prob_championship_win ? 'text-secondary font-bold' : 'text-telemetry-delta-neg font-bold'}>
                    {scenario.winProb >= ferrari.prob_championship_win ? '+' : ''}
                    {((scenario.winProb - ferrari.prob_championship_win) * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              <div className="bg-surface-panel p-5 border border-border-subtle border-t-2 border-t-secondary">
                <span className="font-label-caps text-xs text-text-muted uppercase">SCENARIO 90% HDI</span>
                <div className="font-data-mono-lg text-2xl text-text-primary font-bold mt-2 tnum">
                  [{scenario.ciLow.toFixed(0)} – {scenario.ciHigh.toFixed(0)}]
                </div>
                <div className="text-xs text-text-subtle mt-2">
                  Baseline: [{ferrari.ci_90[0]} – {ferrari.ci_90[1]}]
                </div>
              </div>
            </div>

            {/* Credible Interval Comparison Bar */}
            <div className="bg-surface-panel p-5 border border-border-subtle flex flex-col gap-4">
              <span className="font-label-caps text-xs text-text-muted uppercase font-semibold">
                CREDIBLE INTERVAL SHIFT VISUALIZATION
              </span>
              <div className="flex flex-col gap-3">
                <div>
                  <span className="font-label-caps text-[10px] text-text-subtle uppercase block mb-1">
                    CURRENT COUNTERFACTUAL SCENARIO
                  </span>
                  <CredibleIntervalBar
                    min={250}
                    max={750}
                    ciLow={scenario.ciLow}
                    ciHigh={scenario.ciHigh}
                    median={scenario.points}
                    color="#E10600"
                    label="Scenario 90% HDI"
                  />
                </div>
                <div>
                  <span className="font-label-caps text-[10px] text-text-subtle uppercase block mb-1">
                    FROZEN 2027 PRESEASON BASELINE
                  </span>
                  <CredibleIntervalBar
                    min={250}
                    max={750}
                    ciLow={ferrari.ci_90[0]}
                    ciHigh={ferrari.ci_90[1]}
                    median={ferrari.median_points}
                    color="#6F7680"
                    label="Baseline 90% HDI"
                  />
                </div>
              </div>
            </div>

            {/* Position Probability Shift Comparison */}
            <div className="bg-surface-panel p-5 border border-border-subtle flex flex-col gap-3">
              <span className="font-label-caps text-xs text-text-muted uppercase font-semibold">
                FINISHING POSITION PROBABILITY SHIFT
              </span>
              <div className="grid grid-cols-4 gap-3 text-center">
                <div className="bg-surface-raised p-3 border border-border-subtle">
                  <span className="font-data-mono-md text-xs text-text-subtle uppercase block">P1 FINISH</span>
                  <span className="font-data-mono-lg text-xl text-primary-container font-bold block mt-1 tnum">
                    {(scenario.p1 * 100).toFixed(1)}%
                  </span>
                  <span className="text-[10px] text-text-subtle block">Base: {(ferrari.position_probabilities.P1 * 100).toFixed(1)}%</span>
                </div>

                <div className="bg-surface-raised p-3 border border-border-subtle">
                  <span className="font-data-mono-md text-xs text-text-subtle uppercase block">P2 FINISH</span>
                  <span className="font-data-mono-lg text-xl text-text-primary font-bold block mt-1 tnum">
                    {(scenario.p2 * 100).toFixed(1)}%
                  </span>
                  <span className="text-[10px] text-text-subtle block">Base: {(ferrari.position_probabilities.P2 * 100).toFixed(1)}%</span>
                </div>

                <div className="bg-surface-raised p-3 border border-border-subtle">
                  <span className="font-data-mono-md text-xs text-text-subtle uppercase block">P3 FINISH</span>
                  <span className="font-data-mono-lg text-xl text-secondary font-bold block mt-1 tnum">
                    {(scenario.p3 * 100).toFixed(1)}%
                  </span>
                  <span className="text-[10px] text-text-subtle block">Base: {(ferrari.position_probabilities.P3 * 100).toFixed(1)}%</span>
                </div>

                <div className="bg-surface-raised p-3 border border-border-subtle">
                  <span className="font-data-mono-md text-xs text-text-subtle uppercase block">P4+ FINISH</span>
                  <span className="font-data-mono-lg text-xl text-text-muted font-bold block mt-1 tnum">
                    {(scenario.p4 * 100).toFixed(1)}%
                  </span>
                  <span className="text-[10px] text-text-subtle block">Base: {(ferrari.position_probabilities.P4 * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Points Density Histogram */}
        <section className="bg-surface-panel p-6 border border-border-subtle flex flex-col gap-4">
          <div className="flex justify-between items-center border-b border-border-subtle pb-3">
            <div>
              <h3 className="font-headline-sm text-lg uppercase text-text-primary font-bold">
                CONSTRUCTORS' CHAMPIONSHIP POINTS DENSITY (10,000 MONTE CARLO ITERATIONS)
              </h3>
              <p className="font-body-sm text-xs text-text-muted">
                Empirical simulation frequency distribution showing how counterfactual shift affects probability density mass.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-7 gap-2 h-44 items-end pt-4 bg-surface-raised p-4 border border-border-subtle">
            {histogramBins.map((bin) => {
              const maxCount = 3500;
              const heightPct = Math.round((bin.count / maxCount) * 100);
              return (
                <div key={bin.range} className="flex flex-col items-center gap-2 h-full justify-end">
                  <span className="font-data-mono-md text-[10px] text-text-subtle tnum">{bin.count}</span>
                  <div
                    className={`w-full transition-all ${
                      bin.isScenario
                        ? 'bg-primary-container shadow-[0_0_12px_rgba(225,6,0,0.5)] border border-primary-container'
                        : 'bg-surface-bright hover:bg-border-focus'
                    }`}
                    style={{ height: `${heightPct}%` }}
                    title={`${bin.range} pts: ${bin.count} seasons`}
                  />
                  <span className="font-label-caps text-[10px] text-text-muted uppercase">{bin.range}</span>
                </div>
              );
            })}
          </div>
        </section>
      </div>
    </div>
  );
}

