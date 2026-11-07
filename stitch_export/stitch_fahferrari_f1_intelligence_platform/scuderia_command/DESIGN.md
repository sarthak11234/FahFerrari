---
name: Scuderia Command
colors:
  surface: '#111317'
  surface-dim: '#111317'
  surface-bright: '#37393e'
  surface-container-lowest: '#0c0e12'
  surface-container-low: '#1a1c20'
  surface-container: '#1e2024'
  surface-container-high: '#282a2e'
  surface-container-highest: '#333539'
  on-surface: '#e2e2e8'
  on-surface-variant: '#e9bcb5'
  inverse-surface: '#e2e2e8'
  inverse-on-surface: '#2f3035'
  outline: '#af8781'
  outline-variant: '#5e3f3a'
  surface-tint: '#ffb4a8'
  primary: '#ffb4a8'
  on-primary: '#680200'
  primary-container: '#e10600'
  on-primary-container: '#fff2f0'
  inverse-primary: '#c00500'
  secondary: '#6cdc9f'
  on-secondary: '#003920'
  secondary-container: '#2ea46c'
  on-secondary-container: '#00311b'
  tertiary: '#a8c8ff'
  on-tertiary: '#003062'
  tertiary-container: '#3070c3'
  on-tertiary-container: '#f2f5ff'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#ffdad4'
  primary-fixed-dim: '#ffb4a8'
  on-primary-fixed: '#410100'
  on-primary-fixed-variant: '#930300'
  secondary-fixed: '#89f9b9'
  secondary-fixed-dim: '#6cdc9f'
  on-secondary-fixed: '#002111'
  on-secondary-fixed-variant: '#005231'
  tertiary-fixed: '#d6e3ff'
  tertiary-fixed-dim: '#a8c8ff'
  on-tertiary-fixed: '#001b3c'
  on-tertiary-fixed-variant: '#00468a'
  background: '#111317'
  on-background: '#e2e2e8'
  surface-variant: '#333539'
  canvas-base: '#08090B'
  surface-panel: '#111317'
  surface-raised: '#191C21'
  border-subtle: '#292D33'
  border-focus: '#3E434D'
  text-primary: '#F4F4F2'
  text-muted: '#A0A5AD'
  text-subtle: '#6F7680'
  ferrari-red-hover: '#FF3028'
  telemetry-amber: '#E7B45A'
  telemetry-delta-pos: '#55C58A'
  telemetry-delta-neg: '#E10600'
typography:
  display-xl:
    fontFamily: Barlow Condensed
    fontSize: 56px
    fontWeight: '700'
    lineHeight: 60px
    letterSpacing: 0.02em
  display-xl-mobile:
    fontFamily: Barlow Condensed
    fontSize: 38px
    fontWeight: '700'
    lineHeight: 42px
    letterSpacing: 0.02em
  headline-lg:
    fontFamily: Barlow Condensed
    fontSize: 36px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: 0.03em
  headline-lg-mobile:
    fontFamily: Barlow Condensed
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: 0.03em
  headline-md:
    fontFamily: Barlow Condensed
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: 0.04em
  headline-sm:
    fontFamily: Barlow Condensed
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 22px
    letterSpacing: 0.05em
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 18px
  label-caps:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '600'
    lineHeight: 14px
    letterSpacing: 0.08em
  data-mono-lg:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '700'
    lineHeight: 28px
    letterSpacing: -0.02em
  data-mono-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '600'
    lineHeight: 18px
    letterSpacing: 0.01em
spacing:
  gutter: 1.75rem
  gutter-desktop: 3rem
  gutter-tablet: 1.75rem
  gutter-mobile: 1.125rem
  margin: 1.5rem
  margin-desktop: 3rem
  margin-tablet: 2rem
  margin-mobile: 1rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.875rem
  space-lg: 1.5rem
  space-xl: 2.25rem
---

## Brand & Style

This design system embodies the precision, technical ruthlessness, and mechanical elegance of a top-tier Formula 1 pit wall telemetry suite combined with quantitative financial intelligence. Designed specifically for deep probabilistic forecasting, performance modeling, and competitive benchmarking, the UI rejects generic consumer SaaS cliches—eliminating rounded pastel cards, playful illustrations, neon gradients, and gratuitous blur.

The design movement is **Technical Minimalism and High-Density Motorsport Utility**. It pairs near-black graphite composite surfaces with razor-sharp structural delineation, high-contrast monochrome data typography, and surgical injections of competition red and calibrated telemetry pigments. The target audience comprises motorsport analysts, race strategists, data scientists, and discerning Scuderia enthusiasts who demand mathematical rigor over superficial gamification. The visual language conveys speed through discipline, authority through statistical transparency, and engineering confidence through clean, structured geometry.

## Colors

The chromatic architecture is constructed strictly for technical command environments where low-light readability and cognitive ergonomics are paramount.

- **Canvas & Surface Tiering:** The underlying canvas rests on deep carbon graphite (`#08090B`). Surface modules and operational panels adopt `#111317`, while elevated or active inspector panels step forward to `#191C21`. Structural separation is achieved via razor-thin technical borders (`#292D33`) rather than heavy drop shadows.
- **Brand & Accent:** Ferrari Red (`#E10600`) serves as the singular brand identifier, applied sparingly for primary interactive triggers, active timeline indicators, Ferrari vehicle telemetry, and negative performance deltas. Hover states shift to an energized red (`#FF3028`).
- **Telemetry Semantics:** Telemetry green (`#55C58A`) denotes statistical advantage, positive lap deltas, and valid convergence. Telemetry amber (`#E7B45A`) isolates warning states, high uncertainty intervals, and pit window limits. Telemetry blue (`#6FA8FF`) provides non-competing telemetry overlays, sector timings, and secondary simulation scenarios.
- **Monochrome Readability:** Data clarity is anchored by an off-white primary glyph (`#F4F4F2`), muted descriptor labels (`#A0A5AD`), and structural metadata tags (`#6F7680`), preventing visual fatigue during dense analytical sessions.

## Typography

Typography establishes an immediate dual-role hierarchy:
1. **Command Display (Barlow Condensed):** Headlines, track identifiers, chassis numbers, and section headers use condensed, muscular letterforms set in semi-bold and bold weights with uppercase tracking. This channels pit-wall instrumentation readouts, telemetry consoles, and classic racing broadsheets.
2. **Precision Analytical Body & Data (Inter):** System copy, model methodology, probability values, and table structures rely on Inter. Crucially, all performance metrics, lap deltas, race times, and percentage statistics mandate OpenType tabular numerals (`font-feature-settings: 'tnum' 1, 'zero' 1`). This guarantees that columnar values never jitter during real-time data streaming or recalculations.
3. **Metadata & Status Badges:** Technical subheadings and telemetry categorizations use compact, uppercase micro-typography (`label-caps`) with deliberate character spacing to maximize scanning speed across complex data arrays.

## Layout & Spacing

The layout philosophy mirrors racing telemetry dashboards: structured, disciplined, and strictly bounded.
- **Grid Architecture:** 
  - **Desktop (>= 1440px):** 12-column grid capped at a maximum width of 1440px with 48px (`3rem`) gutters and section margins. Multi-panel dashboards divide cleanly into 3-column metric cards, 6-column distribution histograms, or 9-column trajectory graphs with a 3-column race telemetry sidebar.
  - **Tablet (768px - 1439px):** 8-column grid with 28px (`1.75rem`) gutters, stacking tertiary panels below primary race distribution charts.
  - **Mobile (<= 767px):** 4-column compact flow with 18px (`1.125rem`) gutters. Complex multi-column data tables collapse into swipeable cards or horizontally pinned comparative columns.
- **Rhythm & Padding:** Built on a rigorous 4px baseline. Internal card padding maintains a compact, high-density rhythm (`space-sm` for metric tiles, `space-md` for standard panels, `space-lg` for primary simulation containers). Whitespace is functional rather than decorative, preserved strictly to partition unrelated data domains.

## Elevation & Depth

This design system avoids soft, floating cloud shadows in favor of **Tonal Layering and Low-Contrast Technical Outlines**:
- **Layering System:** Depth is expressed through luminance stepping. The page substrate is `#08090B`. Primary functional modules (telemetry cards, ranking matrices) sit on `#111317`. Hover states, dropdowns, and active tooltips elevate to `#191C21`.
- **Structural Outlines:** Every card, chart container, and panel is outlined with a crisp 1px border (`#292D33`). Under active selection or focus, borders illuminate to `#3E434D` or primary red `#E10600`.
- **Zero Diffuse Shadows:** Floating drop shadows are eliminated. In their place, a razor-thin top highlight line (`inset 0 1px 0 rgba(255, 255, 255, 0.05)`) is etched across elevated containers to replicate machined carbon-fiber and anodized aluminum chassis edges.

## Shapes

The shape geometry is **Sharp (`0`)**. 
- Border radii across cards, KPI modules, buttons, tabs, input fields, and status chips are strictly 0px.
- Sharp rectilinear geometry aligns the UI with carbon chassis panels, pit telemetry screens, tactical flight boards, and trackside timing towers.
- Where clipped status indicators or angular accents occur (such as driver team markers or tab highlights), 45-degree chamfered corners are permissible via CSS polygon masks, further accentuating aerospace motorsport construction.

## Components

### Buttons & Interactive Controls
- **Primary Action:** Solid Ferrari Red (`#E10600`) background, crisp white (`#F4F4F2`) uppercase condensed typography, zero border radius, 0 16px padding. Transitions to `#FF3028` on hover with a micro-scale shift or 1px border flare.
- **Secondary / Ghost Button:** Transparent background, 1px `#292D33` border, `#F4F4F2` text. Hover shifts background to `#191C21` with border color migrating to `#3E434D`.
- **Telemetry Action:** Subdued dark surface with colored left indicator bar (3px width) in green, amber, or blue to toggle specific simulation branches.

### Telemetry Cards & KPI Modules
- Enclosed in `#111317` with a 1px `#292D33` perimeter.
- Top section: Micro-label in uppercase tracking (`#A0A5AD`), accompanied by model provenance or uncertainty metric (`±` whiskers).
- Center: Tabular numeric metric in `data-mono-lg` (`#F4F4F2`).
- Bottom section: Inline delta sparkline or delta chip (`#55C58A` or `#E10600`) indicating pace trend vs. baseline.

### Data Tables & Race Classification
- **Header:** Uppercase `label-caps` in `#6F7680` atop `#08090B` with 1px bottom border.
- **Rows:** Alternating subtle background tint (`transparent` and `rgba(255, 255, 255, 0.015)`). Ferrari entries feature a permanent left accent border in `#E10600` (2px).
- **Cells:** Tabular numbers aligned right, driver codes set in bold display uppercase aligned left.

### Form Inputs & Scenario Sliders
- Rectangular fields bounded by `#292D33`, background `#08090B`, focus ring replaced by 1px solid `#E10600`.
- Range and scenario controls (e.g., simulation weather variables, tire degradation rates) utilize angular slider thumbs on high-contrast mono tracks with dynamic numerical readout callouts.

### Probability Visualizers & Intervals
- Credible intervals are rendered as horizontal error bars with crisp tick-marks at 50%, 80%, and 95% certainty boundaries.
- Heatmaps avoid blurred interpolations, rendering as discretized technical grids with discrete opacity tiers (10%, 25%, 50%, 75%, 100%).