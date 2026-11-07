import React, { useState, useEffect } from 'react';
import { get2027Predictions } from '../../services/api';
import CredibleIntervalBar from '../../components/common/CredibleIntervalBar';
import frozenForecast2027 from '../../data/preseason2027Data.json';
import {
  Flame,
  Shield,
  Gauge,
  Activity,
  Layers,
  Zap,
  TrendingDown,
  TrendingUp,
  Cpu,
  Target,
  Compass,
} from 'lucide-react';

export default function InsideTheScuderiaPage() {
  const [data, setData] = useState(frozenForecast2027);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    async function load() {
      try {
        const preds = await get2027Predictions();
        setData(preds);
      } catch (e) {
        console.error('Failed to load Ferrari analytics:', e);
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
        <span className="font-label-caps text-xs uppercase text-text-muted">Loading Scuderia Telemetry Intelligence...</span>
      </div>
    );
  }

  const ferrari = data.ferrari_constructors_forecast;
  const leclerc = data.driver_forecasts.leclerc;
  const hamilton = data.driver_forecasts.hamilton;

  const archetypes = [
    {
      name: 'High Downforce / Street Circuits',
      circuits: 'Monaco, Singapore (Marina Bay), Hungaroring, Baku',
      winProb: '28.5%',
      expPts: '23.4 pts/race',
      rank: 'P1-P2 Contender',
      description: 'Maximum downforce wing configurations, slow-speed traction, and tight kerb ride compliance maximize SF-27 chassis strengths.',
      color: '#E10600',
    },
    {
      name: 'Low Downforce / High Speed',
      circuits: 'Monza, Spa-Francorchamps, Las Vegas Strip',
      winProb: '25.1%',
      expPts: '21.8 pts/race',
      rank: 'P2-P3 Contender',
      description: 'Low drag aerodynamic package and high straight-line MGU-K battery deployment keep Ferrari competitive against Mercedes PU.',
      color: '#E7B45A',
    },
    {
      name: 'Power & Traction Sensitivity',
      circuits: 'Bahrain (Sakhir), Red Bull Ring, Montreal (Gilles-Villeneuve)',
      winProb: '19.8%',
      expPts: '19.5 pts/race',
      rank: 'P3 Baseline',
      description: 'Rear tire thermal management and turbocharger spool response dictate exit velocity out of low-gear hairpins.',
      color: '#55C58A',
    },
    {
      name: 'Balanced Aero / Lateral G-Load',
      circuits: 'Suzuka, Circuit de Barcelona-Catalunya, Silverstone, COTA',
      winProb: '16.4%',
      expPts: '18.2 pts/race',
      rank: 'P3-P4 Deficit',
      description: 'Sustained high-speed lateral cornering reveals slight understeer deficit relative to Red Bull ground-effect package.',
      color: '#6FA8FF',
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
              SCUDERIA COMMAND · DEEP TELEMETRY
            </span>
          </div>
          <h1 className="font-headline-lg text-headline-lg uppercase text-text-primary font-bold">
            INSIDE THE SCUDERIA
          </h1>
          <p className="font-body-md text-text-muted max-w-3xl leading-relaxed">
            In-depth performance intelligence, aerodynamic archetype sensitivities, driver latent pace decomposition, and discrete-time Weibull reliability hazard modeling for Scuderia Ferrari's 2027 campaign.
          </p>
        </div>
      </section>

      <div className="max-w-[1440px] mx-auto px-4 md:px-margin-desktop py-12 flex flex-col gap-16">
        {/* SECTION 01: 2027 CHAMPIONSHIP OUTLOOK & PERFORMANCE TARGETS */}
        <section className="flex flex-col gap-6">
          <div className="border-b border-border-subtle pb-3">
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-3 bg-primary-container"></span>
              <h2 className="font-headline-md text-headline-md uppercase text-text-primary">
                SECTION 01 // 2027 CHAMPIONSHIP OUTLOOK &amp; PERFORMANCE TARGETS
              </h2>
            </div>
            <p className="font-body-sm text-text-muted mt-0.5">
              Macro season expectations, credible intervals, and target points deficit to benchmark constructors.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-surface-panel p-6 border border-border-subtle border-t-2 border-t-primary-container flex flex-col justify-between">
              <div>
                <span className="font-label-caps text-xs text-text-muted uppercase">SEASON POINT EXPECTATION</span>
                <div className="font-data-mono-lg text-4xl text-text-primary font-bold mt-2 tnum">
                  {ferrari.mean_points.toFixed(1)} PTS
                </div>
                <div className="text-xs text-text-subtle mt-2">
                  Median Baseline: {ferrari.median_points.toFixed(1)} pts
                </div>
              </div>
              <div className="mt-4 pt-4 border-t border-border-subtle">
                <CredibleIntervalBar
                  min={300}
                  max={700}
                  ciLow={ferrari.ci_90[0]}
                  ciHigh={ferrari.ci_90[1]}
                  median={ferrari.median_points}
                  color="#E10600"
                  label="90% HDI [403 – 577 PTS]"
                />
              </div>
            </div>

            <div className="bg-surface-panel p-6 border border-border-subtle border-t-2 border-t-telemetry-amber flex flex-col justify-between">
              <div>
                <span className="font-label-caps text-xs text-text-muted uppercase">BENCHMARK DEFICIT TO P1</span>
                <div className="font-data-mono-lg text-4xl text-telemetry-amber font-bold mt-2 tnum">
                  -181.0 PTS
                </div>
                <div className="text-xs text-text-subtle mt-2">
                  Gap to Mercedes (669.8 pts expected)
                </div>
              </div>
              <div className="mt-4 pt-4 border-t border-border-subtle text-xs text-text-muted">
                Requires aerodynamic upgrade package of approximately <span className="text-text-primary font-semibold">+0.18s per lap</span> to eliminate gap to Mercedes in second half of the season.
              </div>
            </div>

            <div className="bg-surface-panel p-6 border border-border-subtle border-t-2 border-t-secondary flex flex-col justify-between">
              <div>
                <span className="font-label-caps text-xs text-text-muted uppercase">TOP-3 PODIUM RETENTION</span>
                <div className="font-data-mono-lg text-4xl text-secondary font-bold mt-2 tnum">
                  98.5%
                </div>
                <div className="text-xs text-text-subtle mt-2">
                  9,849 of 10,000 paired seasons
                </div>
              </div>
              <div className="mt-4 pt-4 border-t border-border-subtle text-xs text-text-muted">
                Ferrari maintains a comfortable <span className="text-secondary font-semibold">+163.8 pts buffer</span> over 4th-place McLaren, securing a dominant top-3 tier position.
              </div>
            </div>
          </div>
        </section>

        {/* SECTION 02: DRIVER EQUATION · CHARLES LECLERC VS LEWIS HAMILTON */}
        <section className="flex flex-col gap-6">
          <div className="border-b border-border-subtle pb-3">
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-3 bg-primary-container"></span>
              <h2 className="font-headline-md text-headline-md uppercase text-text-primary">
                SECTION 02 // THE DRIVER EQUATION · CHARLES LECLERC VS LEWIS HAMILTON
              </h2>
            </div>
            <p className="font-body-sm text-text-muted mt-0.5">
              Latent pace decomposition &theta;<sub>i,t</sub>, qualifying variance, and simulated intra-team head-to-head outcomes.
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Leclerc Card */}
            <div className="bg-surface-panel p-6 border border-border-subtle border-t-2 border-t-primary-container flex flex-col gap-5 overflow-hidden group">
              <div className="flex justify-between items-start">
                <div className="flex items-center gap-4">
                  <div className="w-16 h-20 bg-surface-raised border border-border-subtle overflow-hidden shrink-0 relative">
                    <img
                      src="/leclerc.jpg"
                      alt="Charles Leclerc"
                      className="w-full h-full object-cover object-top filter grayscale contrast-125 group-hover:filter-none transition-all duration-300"
                      onError={(e) => { e.target.style.display = 'none'; }}
                    />
                  </div>
                  <div>
                    <span className="font-headline-md text-3xl font-bold text-primary-container leading-none">#16</span>
                    <h3 className="font-headline-md text-2xl uppercase text-text-primary font-bold mt-1">
                      CHARLES LECLERC
                    </h3>
                    <span className="font-label-caps text-xs text-text-subtle uppercase">
                      SCUDERIA FERRARI · LEAD QUALIFIER
                    </span>
                  </div>
                </div>
                <div className="text-right">
                  <span className="font-label-caps text-[10px] text-text-subtle uppercase">P(TITLE)</span>
                  <div className="font-data-mono-md text-xl text-primary-container font-bold tnum">
                    {(leclerc.prob_drivers_title * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-3 bg-surface-raised p-3 border border-border-subtle text-xs">
                <div>
                  <span className="font-label-caps text-text-subtle uppercase block text-[10px]">EXP POINTS</span>
                  <span className="font-data-mono-md text-base text-text-primary font-bold tnum">{leclerc.mean_points.toFixed(1)}</span>
                </div>
                <div>
                  <span className="font-label-caps text-text-subtle uppercase block text-[10px]">WINS</span>
                  <span className="font-data-mono-md text-base text-text-primary font-bold tnum">{leclerc.expected_wins.toFixed(1)}</span>
                </div>
                <div>
                  <span className="font-label-caps text-text-subtle uppercase block text-[10px]">PODIUMS</span>
                  <span className="font-data-mono-md text-base text-text-primary font-bold tnum">{leclerc.expected_podiums.toFixed(1)}</span>
                </div>
              </div>

              <div className="flex flex-col gap-3 text-xs">
                <div className="flex justify-between items-center py-1 border-b border-border-subtle/50">
                  <span className="text-text-muted">Qualifying Latent Delta &theta;<sub>q</sub>:</span>
                  <span className="font-data-mono-md text-secondary font-semibold">+0.082s vs teammate</span>
                </div>
                <div className="flex justify-between items-center py-1 border-b border-border-subtle/50">
                  <span className="text-text-muted">Sunday Race-Pace Index:</span>
                  <span className="font-data-mono-md text-text-primary font-semibold">0.912</span>
                </div>
                <div className="flex justify-between items-center py-1">
                  <span className="text-text-muted">Weibull Expected DNFs:</span>
                  <span className="font-data-mono-md text-telemetry-delta-neg font-semibold">{leclerc.expected_dnfs.toFixed(1)} DNFs</span>
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

            {/* Hamilton Card */}
            <div className="bg-surface-panel p-6 border border-border-subtle border-t-2 border-t-telemetry-amber flex flex-col gap-5 overflow-hidden group">
              <div className="flex justify-between items-start">
                <div className="flex items-center gap-4">
                  <div className="w-16 h-20 bg-surface-raised border border-border-subtle overflow-hidden shrink-0 relative">
                    <img
                      src="/hamilton.jpg"
                      alt="Lewis Hamilton"
                      className="w-full h-full object-cover object-top filter grayscale contrast-125 group-hover:filter-none transition-all duration-300"
                      onError={(e) => { e.target.style.display = 'none'; }}
                    />
                  </div>
                  <div>
                    <span className="font-headline-md text-3xl font-bold text-telemetry-amber leading-none">#44</span>
                    <h3 className="font-headline-md text-2xl uppercase text-text-primary font-bold mt-1">
                      LEWIS HAMILTON
                    </h3>
                    <span className="font-label-caps text-xs text-text-subtle uppercase">
                      SCUDERIA FERRARI · RACE EXECUTION
                    </span>
                  </div>
                </div>
                <div className="text-right">
                  <span className="font-label-caps text-[10px] text-text-subtle uppercase">P(TITLE)</span>
                  <div className="font-data-mono-md text-xl text-telemetry-amber font-bold tnum">
                    {(hamilton.prob_drivers_title * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-3 bg-surface-raised p-3 border border-border-subtle text-xs">
                <div>
                  <span className="font-label-caps text-text-subtle uppercase block text-[10px]">EXP POINTS</span>
                  <span className="font-data-mono-md text-base text-text-primary font-bold tnum">{hamilton.mean_points.toFixed(1)}</span>
                </div>
                <div>
                  <span className="font-label-caps text-text-subtle uppercase block text-[10px]">WINS</span>
                  <span className="font-data-mono-md text-base text-text-primary font-bold tnum">{hamilton.expected_wins.toFixed(1)}</span>
                </div>
                <div>
                  <span className="font-label-caps text-text-subtle uppercase block text-[10px]">PODIUMS</span>
                  <span className="font-data-mono-md text-base text-text-primary font-bold tnum">{hamilton.expected_podiums.toFixed(1)}</span>
                </div>
              </div>

              <div className="flex flex-col gap-3 text-xs">
                <div className="flex justify-between items-center py-1 border-b border-border-subtle/50">
                  <span className="text-text-muted">Qualifying Latent Delta &theta;<sub>q</sub>:</span>
                  <span className="font-data-mono-md text-text-muted font-semibold">+0.045s baseline</span>
                </div>
                <div className="flex justify-between items-center py-1 border-b border-border-subtle/50">
                  <span className="text-text-muted">Sunday Race-Pace Index:</span>
                  <span className="font-data-mono-md text-secondary font-semibold">0.931 (Field Top 4)</span>
                </div>
                <div className="flex justify-between items-center py-1">
                  <span className="text-text-muted">Weibull Expected DNFs:</span>
                  <span className="font-data-mono-md text-secondary font-semibold">{hamilton.expected_dnfs.toFixed(1)} DNFs</span>
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
          </div>

          {/* Intra-Team Comparison Analysis Box */}
          <div className="bg-surface-panel p-5 border border-border-subtle text-xs text-text-muted leading-relaxed">
            <span className="font-label-caps uppercase text-text-primary font-bold block mb-1">
              PAIRED CORRELATION DYNAMICS
            </span>
            Under Monte Carlo paired sampling, Hamilton outscores Leclerc in 62.4% of simulated seasons (269.3 pts vs 219.5 pts mean). While Leclerc holds a minor qualifying pace advantage (+0.037s head-to-head median delta), Hamilton's lower DNF hazard rate (1.4 vs 3.0 DNFs) and superior tire degradation retention yield a higher points capture rate across 24 Grands Prix.
          </div>
        </section>

        {/* SECTION 03: CIRCUIT ARCHETYPE SENSITIVITY */}
        <section className="flex flex-col gap-6">
          <div className="border-b border-border-subtle pb-3">
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-3 bg-primary-container"></span>
              <h2 className="font-headline-md text-headline-md uppercase text-text-primary">
                SECTION 03 // CIRCUIT ARCHETYPE SENSITIVITY &amp; AERODYNAMIC PROFILES
              </h2>
            </div>
            <p className="font-body-sm text-text-muted mt-0.5">
              Circuit-level response functions across 4 distinct Formula 1 aerodynamic regimes.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {archetypes.map((a) => (
              <div
                key={a.name}
                className="bg-surface-panel p-6 border border-border-subtle flex flex-col justify-between gap-4"
                style={{ borderTop: `2px solid ${a.color}` }}
              >
                <div>
                  <div className="flex justify-between items-start">
                    <div>
                      <h4 className="font-headline-sm text-lg uppercase text-text-primary font-bold">
                        {a.name}
                      </h4>
                      <span className="font-label-caps text-[10px] text-text-subtle uppercase block mt-0.5">
                        {a.circuits}
                      </span>
                    </div>
                    <span className="px-2 py-0.5 bg-surface-raised border border-border-subtle font-label-caps text-[10px] uppercase font-bold" style={{ color: a.color }}>
                      {a.rank}
                    </span>
                  </div>

                  <p className="text-xs text-text-muted mt-3 leading-relaxed">
                    {a.description}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-3 pt-3 border-t border-border-subtle bg-surface-raised p-2.5 text-xs">
                  <div>
                    <span className="font-label-caps text-[10px] text-text-subtle uppercase block">WIN PROBABILITY</span>
                    <span className="font-data-mono-md text-base text-text-primary font-bold tnum">{a.winProb}</span>
                  </div>
                  <div>
                    <span className="font-label-caps text-[10px] text-text-subtle uppercase block">EXP POINTS / RACE</span>
                    <span className="font-data-mono-md text-base text-text-primary font-bold tnum">{a.expPts}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* SECTION 04: WEIBULL RELIABILITY & DNF HAZARD MODELLING */}
        <section className="bg-surface-panel p-6 border border-border-subtle border-l-4 border-l-telemetry-amber flex flex-col gap-4">
          <div className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-telemetry-amber" />
            <h3 className="font-headline-sm text-lg uppercase text-text-primary font-bold">
              SECTION 04 // WEIBULL RELIABILITY &amp; DNF HAZARD MODELLING (REMEDIATION DEF-03)
            </h3>
          </div>
          <p className="text-xs text-text-muted leading-relaxed">
            In our 2026 model, DNFs were treated as uniform independent coin tosses, resulting in look-ahead bias and failure to capture component fatigue across triple-headers. For 2027, the discrete-time Weibull hazard function &lambda;<sub>i</sub>(t) = (&beta;/&eta;) &middot; (t/&eta;)<sup>&beta;-1</sup> models component lifecycle decay with shape parameter &beta; = 1.35 (wear-out regime) and scale parameter &eta; = 6.2 events.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
            <div className="bg-surface-raised p-3 border border-border-subtle text-xs">
              <span className="font-label-caps text-[10px] text-text-subtle uppercase block">POWER UNIT FATIGUE LIMIT</span>
              <span className="font-data-mono-md text-text-primary font-bold text-base">Rounds 6, 14, 21</span>
              <span className="text-[10px] text-text-subtle block mt-1">Scheduled PU replacement windows</span>
            </div>
            <div className="bg-surface-raised p-3 border border-border-subtle text-xs">
              <span className="font-label-caps text-[10px] text-text-subtle uppercase block">PEAK DNF HAZARD CIRCUIT</span>
              <span className="font-data-mono-md text-telemetry-delta-neg font-bold text-base">Albert Park (14.0%)</span>
              <span className="text-[10px] text-text-subtle block mt-1">High kerb impact & braking wear</span>
            </div>
            <div className="bg-surface-raised p-3 border border-border-subtle text-xs">
              <span className="font-label-caps text-[10px] text-text-subtle uppercase block">SEASON EXPECTED DNF COUNT</span>
              <span className="font-data-mono-md text-text-primary font-bold text-base">4.4 DNFs (Ferrari Total)</span>
              <span className="text-[10px] text-text-subtle block mt-1">Leclerc 3.0 / Hamilton 1.4</span>
            </div>
          </div>
        </section>

        {/* SECTION 05: BENCHMARKING THE RIVALS */}
        <section className="flex flex-col gap-6">
          <div className="border-b border-border-subtle pb-3">
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-3 bg-primary-container"></span>
              <h2 className="font-headline-md text-headline-md uppercase text-text-primary">
                SECTION 05 // BENCHMARKING THE RIVALS · SCUDERIA VS CONTENDERS
              </h2>
            </div>
            <p className="font-body-sm text-text-muted mt-0.5">
              Direct telemetry benchmarking against McLaren Mercedes and Oracle Red Bull Racing.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-surface-panel p-6 border border-border-subtle flex flex-col gap-3">
              <div className="flex justify-between items-center border-b border-border-subtle pb-2">
                <span className="font-headline-sm uppercase text-text-primary font-bold">MCLAREN MERCEDES</span>
                <span className="font-label-caps text-xs text-secondary font-bold">+163.8 PTS BUFFER</span>
              </div>
              <p className="text-xs text-text-muted leading-relaxed">
                McLaren suffers from inconsistent low-speed traction on street layouts and lower mechanical grip relative to Ferrari. Ferrari outqualifies McLaren at 19 of 24 rounds in the simulation baseline.
              </p>
            </div>

            <div className="bg-surface-panel p-6 border border-border-subtle flex flex-col gap-3">
              <div className="flex justify-between items-center border-b border-border-subtle pb-2">
                <span className="font-headline-sm uppercase text-text-primary font-bold">ORACLE RED BULL RACING</span>
                <span className="font-label-caps text-xs text-telemetry-amber font-bold">-78.8 PTS DEFICIT</span>
              </div>
              <p className="text-xs text-text-muted leading-relaxed">
                Red Bull maintains superior aerodynamic efficiency and tire wear degradation in medium and high-speed sectors. Ferrari’s primary opportunity lies in capitalizing on wet races and street circuits where Verstappen’s pace delta narrows.
              </p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
