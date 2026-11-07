import React from 'react';

export default function KpiCard({
  label,
  value,
  subvalue,
  delta,
  deltaType = 'positive', // 'positive' | 'negative' | 'neutral'
  badge,
  accentColor,
  icon: Icon,
}) {
  const deltaColorClass =
    deltaType === 'positive'
      ? 'text-telemetry-delta-pos'
      : deltaType === 'negative'
      ? 'text-telemetry-delta-neg'
      : 'text-telemetry-amber';

  return (
    <div
      className="flex flex-col bg-surface-panel p-space-md border border-border-subtle relative hover:border-border-focus transition-colors"
      style={accentColor ? { borderTop: `2px solid ${accentColor}` } : {}}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="font-label-caps text-label-caps uppercase text-text-muted tracking-wider">
          {label}
        </span>
        {badge && (
          <span className="px-1.5 py-0.5 bg-surface-raised border border-border-subtle font-label-caps text-[10px] uppercase text-text-subtle">
            {badge}
          </span>
        )}
        {Icon && <Icon className="w-4 h-4 text-text-subtle" />}
      </div>

      <div className="font-data-mono-lg text-data-mono-lg text-text-primary tnum font-bold">
        {value}
      </div>

      {(subvalue || delta) && (
        <div className="flex items-center justify-between mt-2 pt-2 border-t border-border-subtle/50 text-xs">
          {subvalue && (
            <span className="text-text-subtle font-body-sm">{subvalue}</span>
          )}
          {delta && (
            <span className={`font-data-mono-md font-semibold tnum ${deltaColorClass}`}>
              {delta}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
