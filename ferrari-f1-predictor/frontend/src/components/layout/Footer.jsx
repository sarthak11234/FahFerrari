import React from 'react';
import { Link } from '../../context/RouterContext';
import { ShieldCheck, GitCommit, Database, ExternalLink } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="w-full bg-canvas-base border-t border-border-subtle py-space-xl text-text-subtle mt-16">
      <div className="max-w-[1440px] mx-auto px-4 md:px-margin-desktop flex flex-col gap-space-lg">
        {/* Top Header Row */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-space-md border-b border-border-subtle pb-space-lg">
          <div className="flex flex-col">
            <div className="flex items-center gap-2">
              <span className="w-2 h-4 bg-primary-container inline-block"></span>
              <span className="font-headline-md text-headline-md uppercase text-text-primary font-bold">
                FAHFERRARI INTELLIGENCE PLATFORM
              </span>
            </div>
            <span className="font-body-sm text-body-sm text-text-muted mt-1">
              Formula 1 Probabilistic Performance Modeling · 2027 Preseason Forecast
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <span className="px-2.5 py-1 bg-surface-panel border border-border-subtle font-label-caps text-[11px] text-text-muted uppercase flex items-center gap-1.5">
              <ShieldCheck className="w-3.5 h-3.5 text-secondary" />
              10,000 MONTE CARLO SEASONS
            </span>
            <span className="px-2.5 py-1 bg-surface-panel border border-border-subtle font-label-caps text-[11px] text-text-muted uppercase flex items-center gap-1.5">
              <GitCommit className="w-3.5 h-3.5 text-primary-container" />
              BASELINE COMMIT: ae91b36
            </span>
            <span className="px-2.5 py-1 bg-surface-panel border border-border-subtle font-label-caps text-[11px] text-text-muted uppercase flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5 text-tertiary" />
              CUTOFF: FEB 18, 2027
            </span>
          </div>
        </div>

        {/* Link Columns Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-space-lg font-body-sm text-body-sm">
          <div className="flex flex-col gap-2">
            <span className="font-label-caps text-label-caps uppercase text-text-primary tracking-wider font-semibold">
              PREDICTIVE MODELS
            </span>
            <Link href="/" className="hover:text-text-primary transition-colors">Overview Forecast</Link>
            <Link href="/championship" className="hover:text-text-primary transition-colors">Championship Matrix</Link>
            <Link href="/ferrari" className="hover:text-text-primary transition-colors">Inside the Scuderia</Link>
            <Link href="/race-calendar" className="hover:text-text-primary transition-colors">24-Round Calendar</Link>
          </div>

          <div className="flex flex-col gap-2">
            <span className="font-label-caps text-label-caps uppercase text-text-primary tracking-wider font-semibold">
              QUANTITATIVE LAB
            </span>
            <Link href="/simulation-lab" className="hover:text-text-primary transition-colors">Simulation Lab</Link>
            <Link href="/model-intelligence" className="hover:text-text-primary transition-colors">Model Intelligence</Link>
            <Link href="/ai-strategist" className="hover:text-text-primary transition-colors">AI Telemetry Strategist</Link>
            <Link href="/2026-retrospective" className="hover:text-text-primary transition-colors">2026 Retrospective & Audit</Link>
          </div>

          <div className="flex flex-col gap-2">
            <span className="font-label-caps text-label-caps uppercase text-text-primary tracking-wider font-semibold">
              CANONICAL DATASETS
            </span>
            <span className="text-text-subtle">F1 Regulation 2026/27 PU</span>
            <span className="text-text-subtle">1,739 Staged Race Records</span>
            <span className="text-text-subtle">Weibull DNF Hazard Functions</span>
            <span className="text-text-subtle">Hierarchical Latent Pace &theta;</span>
          </div>

          <div className="flex flex-col gap-2">
            <span className="font-label-caps text-label-caps uppercase text-text-primary tracking-wider font-semibold">
              INDEPENDENT GOVERNANCE
            </span>
            <p className="text-xs text-text-subtle leading-relaxed">
              FahFerrari is an independent research project. All team names, driver names, circuit names, and trademarks are the property of their respective owners (Formula One Licensing B.V. and Scuderia Ferrari S.p.A.).
            </p>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="pt-space-md border-t border-border-subtle flex flex-col sm:flex-row justify-between items-center gap-2 text-xs font-label-caps uppercase text-text-subtle">
          <span>&copy; 2027 FAHFERRARI RACING INTELLIGENCE. ALL RIGHTS RESERVED.</span>
          <span>ESTIMATED UNCERTAINTIES DERIVED VIA BAYESIAN POSTERIOR SAMPLING · ZERO POINT GUARENTEES</span>
        </div>
      </div>
    </footer>
  );
}

