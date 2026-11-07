import React from 'react';

export default function CredibleIntervalBar({
  min = 0,
  max = 800,
  ciLow,
  ciHigh,
  median,
  color = '#E10600',
  label = '90% HDI',
  showLabels = true,
}) {
  const range = max - min || 1;
  const leftPct = Math.max(0, Math.min(100, ((ciLow - min) / range) * 100));
  const rightPct = Math.max(0, Math.min(100, ((ciHigh - min) / range) * 100));
  const widthPct = Math.max(2, rightPct - leftPct);
  const medianPct = Math.max(0, Math.min(100, ((median - min) / range) * 100));

  return (
    <div className="flex flex-col gap-1 w-full">
      {showLabels && (
        <div className="flex justify-between font-label-caps text-[11px] text-text-subtle tnum">
          <span>{label}: [{ciLow?.toFixed(1)} – {ciHigh?.toFixed(1)}]</span>
          <span>MEDIAN: {median?.toFixed(1)} PTS</span>
        </div>
      )}
      <div className="w-full bg-surface-raised h-2 relative overflow-hidden border border-border-subtle">
        {/* Interval bar */}
        <div
          className="absolute h-full opacity-60"
          style={{
            left: `${leftPct}%`,
            width: `${widthPct}%`,
            backgroundColor: color,
          }}
        />
        {/* Median tick */}
        <div
          className="absolute w-1 h-3 -top-0.5 bg-text-primary z-10"
          style={{
            left: `${medianPct}%`,
          }}
          title={`Median: ${median?.toFixed(1)}`}
        />
      </div>
    </div>
  );
}
