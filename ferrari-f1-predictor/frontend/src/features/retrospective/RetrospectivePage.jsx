import React, { useState, useEffect } from 'react';
import { get2026Predictions, getHistoricalData } from '../../services/api';
import PredictionView from '../../components/legacy/PredictionView';
import Dashboard from '../../components/legacy/Dashboard';
import {
  History,
  AlertTriangle,
  CheckCircle2,
  TrendingDown,
  Layers,
  ArrowRight,
  ShieldAlert,
  Sliders,
  ChevronDown,
  Eye,
} from 'lucide-react';

export default function RetrospectivePage() {
  const [legacyPredictions, setLegacyPredictions] = useState(null);
  const [historicalData, setHistoricalData] = useState([]);
  const [showLegacyRawViewer, setShowLegacyRawViewer] = useState(false);

  useEffect(() => {
    async function loadLegacy() {
      try {
        const [preds, hist] = await Promise.all([
          get2026Predictions(),
          getHistoricalData(),
        ]);
        setLegacyPredictions(preds);
        setHistoricalData(hist);
      } catch (e) {
        console.error(e);
      }
    }
    loadLegacy();
  }, []);

  const macroStats = [
    {
      label: 'CONSTRUCTORS POINTS',
      predicted: '650.0 PTS',
      actual: '460.4 PTS',
      delta: '-189.6 PTS (-29.2%)',
      deltaColor: 'text-telemetry-delta-neg',
      status: 'OVERCONFIDENT BIAS',
    },
    {
      label: 'CHAMPIONSHIP FINISH',
      predicted: 'P1 (CHAMPION)',
      actual: 'P3 (THIRD)',
      delta: '-2 POSITIONS',
      deltaColor: 'text-telemetry-delta-neg',
      status: 'MISCALIBRATED',
    },
    {
      label: '90% CI EMPIRICAL COVERAGE',
      predicted: '[610.0 – 690.0]',
      actual: '460.4 (OUTSIDE CI)',
      delta: '0% VALIDITY',
      deltaColor: 'text-telemetry-delta-neg',
      status: 'PSEUDO-INTERVAL DEF-01',
    },
    {
      label: 'WIN COUNT FORECAST',
      predicted: '11 WINS',
      actual: '5 WINS',
      delta: '-6 WINS',
      deltaColor: 'text-telemetry-delta-neg',
      status: 'OPTIMISM SKEW',
    },
  ];

  const lessons = [
    {
      num: '01',
      title: 'THE PSEUDO-CONFIDENCE INTERVAL FALLACY (DEF-01)',
      problem:
        'The 2026 model manufactured intervals via arbitrary arbitrary normal standard deviations (±40 pts) rather than true posterior uncertainty, creating false certainty and zero ground-truth coverage.',
      remediation:
        '2027 uses 10,000 paired Monte Carlo simulations to extract true empirical 80% and 90% High Density Credible Intervals (HDI).',
    },
    {
      num: '02',
      title: 'RIGID DRIVER MULTIPLIERS & INDEPENDENT MARGINALS (DEF-02)',
      problem:
        'Driver points were modeled as independent marginal distributions and scaled with rigid multipliers, meaning teammates could theoretically score points that violated zero-sum race maximums.',
      remediation:
        '2027 implements joint paired simulation where Leclerc and Hamilton compete in the same simulated races, enforcing strict constructor summation.',
    },
    {
      num: '03',
      title: 'COARSE SEASON AGGREGATIONS & LOSS OF CIRCUIT GRANULARITY (DEF-03)',
      problem:
        'Aggregated full seasons into a single macro model, wiping out circuit sensitivities (street tracks vs low-drag power tracks) and ignoring mechanical component degradation.',
      remediation:
        '2027 uses 4 aerodynamic archetype response functions and a discrete-time Weibull hazard function with shape β = 1.35.',
    },
    {
      num: '04',
      title: 'LOOK-AHEAD BIAS & BACKWARD-FILLING IN DATA PIPELINES (DEF-04)',
      problem:
        'Historical feature tables leaked future season telemetry into early race predictions through backwards rolling fills.',
      remediation:
        '2027 pipeline enforces strict Point-In-Time (PIT) event cutoff boundaries (T_0) with 100% verified zero leakage.',
    },
    {
      num: '05',
      title: 'UNCALIBRATED RANDOM RACE ORDER DISTRIBUTIONS (DEF-05)',
      problem:
        'Uncalibrated order distributions treated tail-end grid finishes as uniform probabilities rather than choice-ranked Plackett-Luce likelihoods.',
      remediation:
        '2027 race outcome engine uses Plackett-Luce choice sampling over latent pace vectors θ_i.',
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
              SCUDERIA COMMAND · HISTORICAL POST-MORTEM
            </span>
          </div>
          <h1 className="font-headline-lg text-headline-lg uppercase text-text-primary font-bold">
            THE 2026 RETROSPECTIVE: THE SEASON THAT TESTED THE MODEL
          </h1>
          <p className="font-body-md text-text-muted max-w-3xl leading-relaxed">
            A forensic engineering review of FahFerrari’s inaugural predictive model—analyzing systematic bias, tracking empirical drift against ground truth, and documenting the Bayesian architecture introduced for 2027.
          </p>
        </div>
      </section>

      <div className="max-w-[1440px] mx-auto px-4 md:px-margin-desktop py-12 flex flex-col gap-16">
        {/* SECTION 1: MACRO ACCURACY & DRIFT */}
        <section className="flex flex-col gap-6">
          <div className="border-b border-border-subtle pb-3">
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-3 bg-primary-container"></span>
              <h2 className="font-headline-md text-headline-md uppercase text-text-primary">
                CHAMPIONSHIP MACRO ACCURACY &amp; PROBABILITY DRIFT
              </h2>
            </div>
            <p className="font-body-sm text-text-muted mt-0.5">
              Comparison between the original 2026 preseason forecast and final FIA championship ground truth.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {macroStats.map((stat) => (
              <div
                key={stat.label}
                className="bg-surface-panel p-5 border border-border-subtle border-t-2 border-t-telemetry-delta-neg flex flex-col justify-between gap-3"
              >
                <div>
                  <span className="font-label-caps text-[10px] text-text-muted uppercase">
                    {stat.label}
                  </span>
                  <div className="grid grid-cols-2 gap-2 mt-2 bg-surface-raised p-2 text-xs border border-border-subtle">
                    <div>
                      <span className="text-[10px] text-text-subtle uppercase block">PREDICTED</span>
                      <span className="font-data-mono-md font-bold text-text-primary">{stat.predicted}</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-text-subtle uppercase block">ACTUAL</span>
                      <span className="font-data-mono-md font-bold text-secondary">{stat.actual}</span>
                    </div>
                  </div>
                </div>

                <div className="flex justify-between items-center text-xs pt-2 border-t border-border-subtle">
                  <span className={`font-data-mono-md font-semibold ${stat.deltaColor}`}>
                    {stat.delta}
                  </span>
                  <span className="font-label-caps text-[10px] text-text-subtle uppercase font-bold">
                    {stat.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* SECTION 2: 5 CRITICAL LESSONS LEARNED */}
        <section className="flex flex-col gap-6">
          <div className="border-b border-border-subtle pb-3">
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-3 bg-primary-container"></span>
              <h2 className="font-headline-md text-headline-md uppercase text-text-primary">
                5 CRITICAL LESSONS LEARNED &amp; BAYESIAN REMEDIATIONS
              </h2>
            </div>
            <p className="font-body-sm text-text-muted mt-0.5">
              Defects identified in our 2026 retrospective and the formal engineering remediations instituted for 2027.
            </p>
          </div>

          <div className="flex flex-col gap-4">
            {lessons.map((lesson) => (
              <div
                key={lesson.num}
                className="bg-surface-panel p-6 border border-border-subtle flex flex-col md:flex-row items-start md:items-center gap-6"
              >
                <div className="font-data-mono-lg text-3xl font-bold text-primary-container shrink-0">
                  {lesson.num}
                </div>
                <div className="flex-1 flex flex-col gap-2">
                  <h3 className="font-headline-sm text-base uppercase text-text-primary font-bold">
                    {lesson.title}
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                    <div className="bg-surface-raised p-3 border border-border-subtle text-text-muted">
                      <span className="font-label-caps text-[10px] text-telemetry-delta-neg uppercase font-bold block mb-1">
                        2026 FAILURE MECHANISM:
                      </span>
                      {lesson.problem}
                    </div>
                    <div className="bg-surface-raised p-3 border border-border-subtle text-text-primary">
                      <span className="font-label-caps text-[10px] text-secondary uppercase font-bold block mb-1">
                        2027 BAYESIAN REMEDIATION:
                      </span>
                      {lesson.remediation}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* SECTION 3: 2026 RAW PREDICTION VIEWER TOGGLE (HISTORICAL PRESERVATION) */}
        <section className="bg-surface-panel p-6 border border-border-subtle flex flex-col gap-4">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-border-subtle pb-4">
            <div>
              <div className="flex items-center gap-2">
                <History className="w-5 h-5 text-text-muted" />
                <h3 className="font-headline-sm text-lg uppercase text-text-primary font-bold">
                  PRESERVED 2026 RAW COMPONENT ARTIFACTS
                </h3>
              </div>
              <p className="text-xs text-text-muted mt-1">
                The original 2026 prediction components remain preserved for complete scientific and engineering accountability.
              </p>
            </div>

            <button
              onClick={() => setShowLegacyRawViewer(!showLegacyRawViewer)}
              className="px-4 py-2 bg-surface-raised border border-border-focus hover:border-primary-container text-text-primary font-label-caps text-xs uppercase transition-colors flex items-center gap-2"
            >
              <Eye className="w-4 h-4 text-primary-container" />
              <span>{showLegacyRawViewer ? 'Hide Original 2026 UI' : 'View Original 2026 Raw UI'}</span>
            </button>
          </div>

          {showLegacyRawViewer && (
            <div className="pt-4 border-t border-border-subtle">
              <div className="bg-canvas-base p-4 border border-border-subtle mb-6 text-xs text-telemetry-amber font-mono">
                // ACTIVE: MOUNTING PRESERVED 2026 LEGACY PREDICTION VIEW (React 18 Component Stack)
              </div>
              {legacyPredictions ? (
                <div className="flex flex-col gap-8">
                  <PredictionView predictions={legacyPredictions} />
                  {historicalData.length > 0 && <Dashboard data={historicalData} />}
                </div>
              ) : (
                <div className="p-6 text-center text-xs text-text-subtle">
                  Loading preserved 2026 predictor state...
                </div>
              )}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

