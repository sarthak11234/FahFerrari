import React, { useState } from 'react';
import { useRouter, Link } from '../../context/RouterContext';
import { Terminal, Menu, X, ChevronDown } from 'lucide-react';

export default function Navbar() {
  const { currentPath, navigate, season, setSeason } = useRouter();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [seasonDropdownOpen, setSeasonDropdownOpen] = useState(false);

  const navLinks = [
    { name: 'Overview', path: '/' },
    { name: 'Championship', path: '/championship' },
    { name: 'Ferrari', path: '/ferrari' },
    { name: 'Race Calendar', path: '/race-calendar' },
    { name: 'Simulation Lab', path: '/simulation-lab' },
    { name: 'Model Intelligence', path: '/model-intelligence' },
    { name: 'AI Strategist', path: '/ai-strategist' },
    { name: '2026 Retrospective', path: '/2026-retrospective' },
  ];

  const handleSeasonSelect = (newSeason) => {
    setSeason(newSeason);
    setSeasonDropdownOpen(false);
    if (newSeason === '2026' && currentPath !== '/2026-retrospective') {
      navigate('/2026-retrospective');
    } else if (newSeason === '2027' && currentPath === '/2026-retrospective') {
      navigate('/');
    }
  };

  return (
    <header className="fixed top-0 left-0 w-full z-50 bg-canvas-base/95 backdrop-blur-md border-b border-border-subtle">
      <div className="h-16 w-full max-w-[1440px] mx-auto px-4 md:px-margin-desktop flex items-center justify-between">
        {/* Brand / Logo */}
        <Link href="/" className="flex items-center gap-space-sm group">
          <div className="w-2.5 h-6 bg-primary-container group-hover:bg-ferrari-red-hover transition-colors shrink-0"></div>
          <div className="flex flex-col">
            <span className="font-headline-md text-headline-md tracking-wider uppercase text-text-primary group-hover:text-primary-container transition-colors font-bold leading-tight">
              FAHFERRARI
            </span>
            <span className="font-label-caps text-[10px] tracking-widest text-text-muted uppercase">
              F1 PERFORMANCE INTELLIGENCE
            </span>
          </div>
        </Link>

        {/* Desktop Navigation Links */}
        <nav className="hidden xl:flex items-center space-x-6 h-full">
          {navLinks.map((link) => {
            const isActive = currentPath === link.path;
            return (
              <Link
                key={link.path}
                href={link.path}
                className={`font-label-caps text-label-caps uppercase tracking-wider transition-colors pt-[22px] pb-[22px] border-b-2 ${
                  isActive
                    ? 'text-text-primary border-primary-container font-bold'
                    : 'text-text-muted hover:text-text-primary border-transparent'
                }`}
              >
                {link.name}
              </Link>
            );
          })}
        </nav>

        {/* Right Status & Season Actions */}
        <div className="flex items-center gap-space-sm shrink-0">
          {/* Season Selector */}
          <div className="relative">
            <button
              onClick={() => setSeasonDropdownOpen(!seasonDropdownOpen)}
              className="flex items-center bg-surface-panel border border-border-subtle hover:border-border-focus px-space-sm py-1 transition-colors text-left"
              type="button"
            >
              <span className="font-data-mono-md text-body-sm text-text-muted tracking-wider uppercase mr-1">
                SEASON
              </span>
              <span className="font-data-mono-md text-body-sm text-text-primary uppercase font-bold">
                {season === '2027' ? '2027 PRESEASON' : '2026 RETROSPECTIVE'}
              </span>
              <ChevronDown className="w-4 h-4 text-text-muted ml-1" />
            </button>

            {seasonDropdownOpen && (
              <div className="absolute right-0 mt-1 w-52 bg-surface-raised border border-border-focus shadow-2xl z-50">
                <button
                  onClick={() => handleSeasonSelect('2027')}
                  className={`w-full text-left px-3 py-2 text-xs font-label-caps uppercase flex items-center justify-between transition-colors ${
                    season === '2027' ? 'bg-primary-container/20 text-primary-container font-bold border-l-2 border-primary-container' : 'text-text-primary hover:bg-surface-panel'
                  }`}
                >
                  <span>2027 Preseason Forecast</span>
                  <span className="text-[10px] bg-secondary-container/20 text-secondary px-1 py-0.5">FROZEN</span>
                </button>
                <button
                  onClick={() => handleSeasonSelect('2026')}
                  className={`w-full text-left px-3 py-2 text-xs font-label-caps uppercase flex items-center justify-between transition-colors ${
                    season === '2026' ? 'bg-primary-container/20 text-primary-container font-bold border-l-2 border-primary-container' : 'text-text-primary hover:bg-surface-panel'
                  }`}
                >
                  <span>2026 Retrospective Baseline</span>
                  <span className="text-[10px] bg-surface-panel text-text-muted px-1 py-0.5">AUDITED</span>
                </button>
              </div>
            )}
          </div>

          {/* Preseason Snapshot Frozen Badge */}
          <div className="hidden lg:flex items-center gap-space-xs bg-surface-panel border border-border-subtle px-space-sm py-1">
            <span className="w-1.5 h-1.5 rounded-full bg-secondary shadow-[0_0_8px_#55C58A] animate-pulse status-dot"></span>
            <span className="font-label-caps text-label-caps uppercase text-text-primary">
              PRESEASON SNAPSHOT · FROZEN
            </span>
          </div>

          {/* Data Cutoff Badge */}
          <div className="hidden 2xl:block bg-surface-panel border border-border-subtle px-space-sm py-1 font-label-caps text-label-caps uppercase text-text-subtle">
            DATA CUTOFF: FEB 18, 2027
          </div>

          {/* Terminal / Strategist Quick Trigger */}
          <Link
            href="/ai-strategist"
            className="p-1.5 bg-surface-panel border border-border-subtle hover:border-primary-container hover:bg-surface-raised text-text-muted hover:text-text-primary transition-colors flex items-center justify-center"
            title="Launch AI Strategist Telemetry Terminal"
          >
            <Terminal className="w-4 h-4 text-primary-container" />
          </Link>

          {/* Mobile Hamburger Menu */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="xl:hidden p-1.5 bg-surface-panel border border-border-subtle text-text-muted hover:text-text-primary"
            aria-label="Toggle navigation"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Navigation Drawer */}
      {mobileMenuOpen && (
        <div className="xl:hidden bg-surface-panel border-b border-border-focus px-4 py-3 flex flex-col space-y-2">
          {navLinks.map((link) => {
            const isActive = currentPath === link.path;
            return (
              <Link
                key={link.path}
                href={link.path}
                onClick={() => setMobileMenuOpen(false)}
                className={`py-2 px-3 text-xs font-label-caps uppercase tracking-wider transition-colors flex items-center justify-between ${
                  isActive
                    ? 'bg-primary-container/20 text-primary-container border-l-2 border-primary-container font-bold'
                    : 'text-text-muted hover:text-text-primary'
                }`}
              >
                <span>{link.name}</span>
                {isActive && <span className="w-1.5 h-1.5 bg-primary-container"></span>}
              </Link>
            );
          })}
          <div className="pt-2 border-t border-border-subtle flex flex-col gap-1 text-[11px] text-text-subtle font-label-caps uppercase">
            <span>PRESEASON SNAPSHOT · FROZEN (10,000 MC RUNS)</span>
            <span>DATA CUTOFF: FEB 18, 2027</span>
          </div>
        </div>
      )}
    </header>
  );
}

