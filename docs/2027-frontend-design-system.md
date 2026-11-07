# FahFerrari 2027 — Frontend Design System & Homepage Plan

**Design direction:** Scuderia Command  
**Density:** Balanced — editorial storytelling with high-value analytics  
**Homepage hero:** Cinematic Ferrari visual  
**Motion:** Dynamic, motorsport-inspired, performance-conscious  
**Frontend baseline:** Existing React 18 + Vite + Axios + Recharts + Lucide  
**Backend integration:** Flask API under `/api`

---

## 1. Product design intent

FahFerrari should feel like a premium motorsport intelligence product: part Ferrari performance briefing, part race-engineering workstation, part statistical forecasting platform.

The interface must communicate three things immediately:
1. This is a Ferrari-focused F1 forecasting product.
2. The predictions are generated from a full-grid, probabilistic simulation—not manually typed rankings.
3. Every forecast has uncertainty, provenance and a path to inspect the underlying evidence.

Avoid a generic admin-dashboard appearance. Use strong editorial hierarchy, large motorsport imagery, restrained Ferrari red, precise chart typography and deliberate negative space.

## 2. Brand and visual identity

### Brand attributes
- Confident, technical, cinematic, precise, analytical.
- Ferrari red is the signature accent, not the background for every component.
- Competitor teams retain distinct but muted identity cues in comparative charts.
- Data visualisations should prioritise legibility over decorative effects.

### Colour tokens

| Token | Hex | Usage |
|---|---|---|
| `--ff-bg` | `#08090B` | Main application background |
| `--ff-surface` | `#111317` | Cards and panels |
| `--ff-surface-raised` | `#191C21` | Menus, hover surfaces |
| `--ff-border` | `#292D33` | Subtle dividers and card borders |
| `--ff-text` | `#F4F4F2` | Primary text |
| `--ff-muted` | `#A0A5AD` | Secondary text |
| `--ff-subtle` | `#6F7680` | Metadata and axis labels |
| `--ff-red` | `#E10600` | Ferrari accent / active state |
| `--ff-red-hover` | `#FF3028` | Hover and focus accent |
| `--ff-green` | `#55C58A` | Positive status only |
| `--ff-amber` | `#E7B45A` | Warning / uncertainty attention |
| `--ff-blue` | `#6FA8FF` | Neutral analytical comparison |

Do not encode meaning using colour alone. Pair statuses with labels, icons or patterns. Check text and chart contrast against WCAG AA.

### Typography

Use a two-family system:
- Display/headlines: **Barlow Condensed** (700/800) or **Archivo Narrow**. Use for large titles, race labels and numeric emphasis.
- UI/body/data: **Inter** (400/500/600). Use for navigation, descriptions, controls, chart labels and tables.

Fallback stacks must be system-safe. Use tabular numerals for points, probabilities, lap times and dates. Avoid using condensed type for long explanatory text.

Suggested scale:
- Display hero: clamp(3rem, 7vw, 6.5rem), tight line-height.
- Page title: 2.25rem–3rem.
- Section heading: 1.25rem–1.5rem.
- KPI value: 2rem–3rem.
- Body: 0.9rem–1rem.
- Metadata: 0.7rem–0.8rem.

### Layout and spacing

- Desktop content max-width: 1440px; wide dashboard sections may use 1600px.
- Main page gutters: 48px desktop, 28px tablet, 18px mobile.
- 12-column desktop grid; 8-column tablet; 4-column mobile.
- Base spacing unit: 4px; common gaps 8, 12, 16, 24, 32, 48, 64.
- Card radius: 12px; larger feature panels: 18px.
- Borders are thin and low contrast. Avoid excessive glassmorphism.
- Keep page rhythm: cinematic hero → compact forecast summary → analytical sections → provenance/footer.

### Surfaces and elevation

Use opaque graphite surfaces as the default. Reserve gradients for hero overlays and selected featured elements. Avoid bright glows around every card. Use a subtle inner border and restrained shadow for raised overlays only.

## 3. Navigation and information architecture

### Primary desktop navigation

Left: FahFerrari mark and wordmark.  
Centre: Overview, Championship, Ferrari, Race Calendar, Simulation Lab, Model Intelligence.  
Right: season/version selector, API status indicator, profile/settings if required.

AI Strategist can be a distinct utility entry or a persistent floating entry point, but should not compete with the main navigation.

### Page map

- Overview — command centre / homepage.
- Championship — constructors and driver championship distributions.
- Ferrari — Hamilton vs Leclerc, Ferrari team performance, circuit performance.
- Race Calendar — 24 event forecasts and Sprint markers.
- Simulation Lab — Monte Carlo explorer and what-if scenarios.
- Model Intelligence — architecture, dataset, validation, provenance.
- AI Strategist — conversational access to forecast context.
- Retrospective — 2026 prediction versus actual performance.

### Responsive navigation

Desktop uses a horizontal navigation bar. Tablet may collapse secondary links into a menu. Mobile uses a compact header and accessible drawer or bottom navigation for the most-used sections. Never shrink the full desktop nav until labels become unreadable.

## 4. Component design language

### Buttons
- Primary: Ferrari red fill, white label, clear focus ring.
- Secondary: graphite fill with subtle border.
- Tertiary: text-only, used for low-priority actions.
- Destructive: reserved for destructive actions, not ordinary forecast controls.
- Minimum touch target: 44×44px.

### Cards
- Standard analytical card: surface background, 1px border, 12px radius, 20–24px padding.
- Featured card: larger radius and subtle red accent edge or eyebrow label.
- Avoid nesting cards inside cards unless the hierarchy requires it.

### KPI cards
Each KPI includes a descriptive label, main value, context/interval and optional trend. Example:
- Expected points: 488.8
- Median: 488.0
- 90% interval: 403–577

Use the actual API values at runtime. These figures are examples from the provided API contract, not approved hardcoded UI copy.

### Tables
- Sticky header where useful.
- Right-align points and probability columns.
- Use team identity markers but retain readable text labels.
- Ferrari row receives a subtle red left border / tint.
- Provide responsive card conversion or horizontal scroll on narrow screens.

### Tooltips
Explain statistical terms in plain language:
- Mean vs median.
- 80% / 90% credible interval.
- Championship win probability.
- Expected wins / podiums.
- DNF risk.
- Simulation count and random seed.

### Empty, loading and error states
- Loading: skeleton blocks that preserve layout, not a full-screen spinner after initial shell render.
- No data: explain what source or endpoint is missing and provide retry.
- API failure: show last successful snapshot only if timestamped and clearly labelled stale.
- Partial data: show available panels and identify unavailable data; do not fabricate zeros.
- Never label an unconfirmed or illustrative forecast as official.

## 5. Homepage blueprint — Overview / Command Centre

### A. Global header
- FahFerrari logo/wordmark.
- Main navigation.
- Season selector: 2027 Forecast / 2026 Retrospective.
- Forecast state label: Preseason / Updated after Round N / Archived.
- Compact API health indicator.

Header height target: 72–80px desktop; compact on mobile. Use a translucent graphite background only if it maintains contrast over the hero.

### B. Cinematic hero

Composition:
- Full-width 16:9 or 2.2:1 visual region, 480–620px desktop height.
- Full-bleed Ferrari race-car image with subject placed to one side, leaving negative space for copy.
- Dark gradient overlay from the text side; preserve image detail.
- Small eyebrow: `F1 PERFORMANCE INTELLIGENCE · 2027`.
- Headline: `THE 2027 SEASON.` then `PREDICTED BY DATA.`
- Supporting line: `24 Grands Prix. 10 Sprint events. 10,000 simulated seasons.` only when confirmed by current API metadata.
- Primary CTA: `Explore the forecast`.
- Secondary CTA: `How the model works`.
- Compact metadata: model version, forecast cutoff and status from provenance endpoint.

The hero image must be replaceable via a local asset or configured CDN path. Do not make the application dependent on a remote search-image URL at runtime. Confirm image licensing before release.

### C. Season-at-a-glance strip

Three or four compact indicators:
- Simulated seasons.
- Grand Prix count.
- Sprint event count.
- Regulation era / forecast version.

These values come from `simulation_meta` and `/api/predictions/2027/provenance`. If provenance is unavailable, show an unavailable state rather than inventing a value.

### D. Ferrari championship outlook

Featured wide card:
- Ferrari expected constructors' points.
- Median points.
- 90% credible interval.
- Championship P1 probability.
- Top-three probability.
- Most likely finishing position.

Visual: interval/range bar with a marker for median and mean. Label the interval explicitly as 90%; do not present it as a guaranteed range.

### E. Driver battle — Hamilton vs Leclerc

Two balanced driver panels with:
- Mean and median points.
- 90% interval.
- Expected wins, podiums and DNFs.
- Drivers' title probability.
- Position probability distribution.

Add a compact head-to-head chart only if the backend exposes paired simulation outcomes. Marginal driver distributions alone cannot establish the probability that one teammate finishes ahead of the other.

### F. Constructors' field

A ranked 10-team table:
- Expected position.
- Constructor.
- Mean points.
- 90% interval.
- P1 probability.
- Top-three probability.

Highlight Ferrari without visually suppressing competitors. Add a link to the full Championship page.

### G. Race watchlist

Show selected high-interest events or next races, not all 24 cards in the homepage. Each preview contains:
- Round and circuit.
- Sprint indicator.
- Expected Ferrari points.
- Win / podium probability.
- DNF risk.

The detailed calendar page contains all events. The current sample API response contains only selected rounds; do not assume it contains all 24. Confirm the full endpoint contract before designing a complete calendar from this payload.

### H. Model integrity / provenance footer panel

Show:
- Model architecture summary.
- Simulation count and seed.
- Forecast cutoff timestamp.
- Regulation era.
- Data/model version.
- Link to Model Intelligence.

Use `/api/predictions/2027/provenance` as the source of truth.

### I. Footer
- FahFerrari identity and project description.
- Data-source acknowledgements.
- Model limitations / methodology link.
- GitHub repository link.
- Forecast version and timestamp.

## 6. API integration and frontend data contracts

The supplied backend reference identifies these endpoints:

| Endpoint | Frontend use |
|---|---|
| `GET /api/predictions/2027` | Primary forecast |
| `GET /api/predictions/2027/provenance` | Forecast freeze / model metadata |
| `GET /api/predictions/2026` | Legacy comparison |
| `GET /api/historical` | Historical trend and retrospective |
| `POST /api/chat` | AI Strategist |

The existing React app currently fetches `/api/historical` and `/api/predictions/2026` directly in `App.jsx`. The package uses React 18, Vite, Axios, Recharts and Lucide. Keep this baseline and refactor incrementally.

### Suggested frontend data layer

- Centralise API base URL in environment configuration, e.g. `VITE_API_BASE_URL`.
- Create a typed/validated API service module even if the project remains JavaScript.
- Separate network state from presentation components.
- Fetch forecast and provenance concurrently; handle independent failure states.
- Normalise constructor/driver IDs to display labels in a presentation mapping layer.
- Preserve raw API response fields; do not mutate source data for chart convenience.
- Use explicit loading, error, stale and partial states.
- Avoid hardcoding example response numbers into components.

Suggested modules:
```text
frontend/src/
  app/
    App.jsx
    routes.js
  components/
    layout/
      AppShell.jsx
      Header.jsx
      Footer.jsx
      MobileNavigation.jsx
    ui/
      Button.jsx
      Card.jsx
      Badge.jsx
      Tooltip.jsx
      Skeleton.jsx
      EmptyState.jsx
    forecast/
      ForecastHero.jsx
      SeasonMetaStrip.jsx
      FerrariOutlook.jsx
      CredibleInterval.jsx
      DriverBattlePreview.jsx
      ConstructorsTable.jsx
      RaceWatchlist.jsx
      ProvenancePanel.jsx
  pages/
    OverviewPage.jsx
    ChampionshipPage.jsx
    FerrariPage.jsx
    CalendarPage.jsx
    SimulationLabPage.jsx
    ModelIntelligencePage.jsx
    StrategistPage.jsx
    RetrospectivePage.jsx
  services/
    apiClient.js
    predictions.js
    historical.js
    strategist.js
  hooks/
    useForecast.js
    useProvenance.js
  utils/
    formatters.js
    constructorDisplay.js
  styles/
    tokens.css
    globals.css
    motion.css
  assets/
    hero/
```

This is a target structure, not a requirement to create all files in one pass. Inspect current files and reuse components where sensible.

## 7. Motion and interaction system

Motion should feel like a racing broadcast package: purposeful, directional and quick. Avoid constant movement or distracting looping animations.

### Motion principles
- Page entrance: staggered opacity + 8–16px vertical translation, 350–550ms.
- KPI reveal: count-up once when data first arrives; respect reduced-motion preference.
- Chart reveal: animate series once, not on every rerender.
- Navigation: 150–220ms underline/colour transition.
- Card hover: 2–3px lift, border accent and restrained shadow.
- Hero: subtle parallax or slow image scale only on capable desktop devices; disable on mobile/reduced-motion.
- Loading: shimmer skeleton with low contrast.
- Modal/drawer: short opacity + transform transition.

### Accessibility and performance
- Respect `prefers-reduced-motion`.
- Never animate flashing red or high-frequency telemetry-like effects.
- Avoid scroll-jacking.
- Avoid heavy video backgrounds until performance and mobile fallback are tested.
- Lazy-load below-the-fold images and charts.
- Keep chart animations short and disable them for reduced motion.

## 8. Chart specification

- Championship points: horizontal bar chart with interval whiskers.
- Points distribution: histogram / density approximation if raw simulation samples are available.
- Position probabilities: horizontal probability bars or heatmap.
- Driver comparison: paired distributions only if paired simulation data is exposed.
- Race-by-race: line or bar chart with credible interval bands.
- Circuit matrix: circuit × metric heatmap with text labels and accessible legend.

Use Recharts for standard charts already supported by the installed dependency. Add D3 only for a visualisation that Recharts cannot represent cleanly. All charts need axis labels, units, tooltips and a text/table alternative.

## 9. Implementation sequence

### UI-0 — Audit and contract verification
- Inspect all current frontend components and CSS.
- Verify actual Flask response payloads against the supplied contract.
- Confirm whether all 24 races and all 20 drivers are returned.
- Confirm whether raw Monte Carlo samples or only aggregates are available.
- Confirm forecast cutoff and stale-data behaviour.

**Exit:** frontend contract document matches running backend responses.

### UI-1 — Design foundation
- Add design tokens, typography, global reset and responsive container.
- Establish AppShell, navigation, cards, buttons, badges, tooltips and skeletons.
- Add reduced-motion and keyboard focus standards.

**Exit:** responsive shell and reusable components approved.

### UI-2 — Cinematic homepage hero
- Add licensed/local hero asset and responsive crop.
- Build hero copy, forecast state, CTA and provenance metadata.
- Implement restrained dynamic entrance effects.

**Exit:** hero works across desktop/tablet/mobile and has a static fallback.

### UI-3 — Forecast data integration
- Centralise API client and environment configuration.
- Fetch 2027 forecast and provenance.
- Implement loading/error/partial/stale states.
- Add runtime response validation.

**Exit:** page renders live API data and handles backend downtime safely.

### UI-4 — Homepage analytics
- Implement season metadata, Ferrari outlook, driver panels, constructor table, race watchlist and provenance panel.
- Add charts and tooltips.
- Verify displayed totals/intervals are derived from API values.

**Exit:** homepage supports the complete forecast contract without hardcoded predictions.

### UI-5 — Responsive and quality pass
- Test keyboard navigation, contrast, reduced motion, mobile layouts and chart legibility.
- Test slow API, empty response, malformed payload and partial failure.
- Check performance and image loading.

**Exit:** production-quality overview page and documented outstanding issues.

## 10. Acceptance criteria for homepage

1. The page visually expresses the Scuderia Command identity.
2. The hero is cinematic but headline and CTA remain readable at every breakpoint.
3. All numerical forecast values are sourced from the API.
4. Mean, median, credible interval and probabilities are labelled correctly.
5. No missing field is silently converted to a misleading zero.
6. Forecast cutoff, version and simulation metadata are visible.
7. The user can navigate to championship, Ferrari analysis, calendar and methodology.
8. Loading, error, stale and partial-data states are designed.
9. Keyboard focus and reduced-motion preferences work.
10. The page is usable at 360px mobile width and large desktop widths.
11. No claim implies that an illustrative payload is a verified live forecast.
12. Existing 2026 functionality remains accessible.

## 11. Important data-contract caveats

The supplied API reference is the design basis, but it is not a substitute for checking the running backend. Specifically:
- The race-by-race example lists only a handful of rounds, not all 24.
- The example constructor and driver forecasts should be treated as example values until verified against the active forecast artifact.
- A paired teammate head-to-head probability requires paired simulation outcomes, not only separate marginal distributions.
- A full probability histogram requires simulation samples or precomputed bins; means and credible intervals alone are insufficient.
- The exact 2027 scoring rules and regulations must be verified from current official documents and the backend's versioned rules before labels are finalised.

## 12. Definition of done

The first frontend milestone is complete when the Overview page is a responsive, accessible, cinematic command centre that renders verified 2027 forecast and provenance data, communicates uncertainty accurately, survives API failure, and preserves the 2026 retrospective path.

The target is not to build every navigation destination immediately. Build the design foundation and Overview page first, then extend the system consistently to the remaining sections.
