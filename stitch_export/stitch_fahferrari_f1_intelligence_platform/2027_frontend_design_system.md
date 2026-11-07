# FahFerrari 2027 — Frontend Design System
**Design direction:** Scuderia Command  
**Status:** Stitch design reference reconstructed from the approved project brief.

## Product
FahFerrari is an independent, Ferrari-focused Formula 1 performance-intelligence platform. It forecasts Ferrari's 2027 campaign while benchmarking all constructors and drivers. The interface must communicate forecast outcomes, modelling rationale, and uncertainty—not simply display sports scores.

## Visual identity
- Premium motorsport engineering command centre with editorial storytelling.
- Near-black graphite surfaces, Ferrari red accents, crisp typography, restrained telemetry colours.
- Avoid generic SaaS dashboards, excessive rounded cards, neon gradients, gratuitous glassmorphism, and decorative charts.

## Tokens
| Token | Value |
|---|---|
| Background | `#08090B` |
| Surface | `#111317` |
| Raised surface | `#191C21` |
| Border | `#292D33` |
| Primary text | `#F4F4F2` |
| Muted text | `#A0A5AD` |
| Subtle text | `#6F7680` |
| Ferrari red | `#E10600` |
| Red hover | `#FF3028` |
| Telemetry green | `#55C58A` |
| Amber | `#E7B45A` |
| Blue | `#6FA8FF` |

## Typography
- Display/headlines: Barlow Condensed or Archivo Narrow.
- Body/UI: Inter.
- Use tabular numerals for statistics, percentages, and points.
- Strong editorial headlines; compact but readable data labels.

## Layout
- 12-column desktop grid; 8-column tablet; 4-column mobile.
- Maximum content width approximately 1440px.
- Gutters: 48px desktop, 28px tablet, 18px mobile.
- 4px spacing unit.
- Responsive targets: 1440px, 1280px, 768px, 360px.
- Keep charts legible; provide mobile-friendly alternatives to wide tables.

## Motion
- Staggered entrance: approximately 350–550ms.
- KPI count-up once; chart reveal once.
- Hover elevation 2–3px.
- Subtle hero parallax/scale on desktop only.
- Respect `prefers-reduced-motion`.
- No flashing, rapid loops, or continuously changing numbers.

## Global navigation
FahFerrari wordmark and subtitle `F1 PERFORMANCE INTELLIGENCE`.
Navigation: Overview, Championship, Ferrari, Race Calendar, Simulation Lab, Model Intelligence.
Season selector: 2027. Show forecast status from provenance data. Mobile navigation must remain accessible.

## Homepage structure
1. Cinematic Ferrari hero (480–620px desktop; 16:9 or approximately 2.2:1).
   - Eyebrow: `F1 PERFORMANCE INTELLIGENCE · 2027`
   - Headline: `THE 2027 SEASON. PREDICTED BY DATA.`
   - Explain probabilistic Ferrari forecasting.
   - CTAs: `EXPLORE THE FORECAST`, `HOW THE MODEL WORKS`.
   - Display model version, cutoff, and status from the API.
2. Season metadata strip: GP count, Sprint count, simulations, regulation era—API-driven.
3. Ferrari constructor outlook: mean, median, 90% interval, P1/top-three probabilities, likely position.
4. Driver battle: Charles Leclerc and Lewis Hamilton; points, intervals, wins, podiums, DNFs, title probabilities.
5. Constructors field: all teams, expected points, intervals, P1/top-three probabilities; Ferrari highlighted, rivals not suppressed.
6. Race watchlist: selected events with only supported forecast fields.
7. Model integrity/provenance panel.
8. Footer: sources, limitations, timestamp, GitHub.

## Application pages
- Overview: season narrative and Ferrari outlook.
- Championship: drivers/constructors rankings, distributions, intervals, probability heatmaps.
- Ferrari: team outlook, driver comparison, circuit and race performance.
- Race Calendar: round, Grand Prix, circuit, date, sprint flag, forecast status.
- Simulation Lab: distributions and hypothetical scenario controls, only where backend supports them.
- Model Intelligence: architecture, data coverage, validation, provenance.
- AI Strategist: conversation interface that distinguishes model facts from AI interpretation.
- 2026 Retrospective: original predictions versus actual results and lessons learned.

## Reusable components
Navigation, hero, KPI cards, probability cards, driver comparison, constructor table, chart panels, race cards, filters, tooltips, status indicators, skeletons, empty/error states, footer.

## Chart language
- Championship ranking: horizontal bars with interval whiskers.
- Points distribution: histogram/density only if samples or bins exist.
- Position probabilities: bars or heatmap.
- Race-by-race: line/bar with uncertainty intervals.
- Circuit performance: heatmap only when supported by data.
- Use tooltips to explain mean, median, intervals, and probabilities.

## Data integrity and UI states
All displayed numbers must be API-sourced. Never silently turn missing values into zero. Clearly distinguish forecast from actual results, expected values from medians, and credible intervals from prediction intervals. Do not derive teammate head-to-head probabilities from independent marginal distributions. Design loading, error, missing, partial, stale, and forecast-not-yet-available states.

## Brand and imagery
Use a cinematic Ferrari F1 image with deliberate negative space for hero text. Use Ferrari branding assets only where rights permit. Do not invent team logos or imply official Ferrari affiliation. FahFerrari is an independent project.
