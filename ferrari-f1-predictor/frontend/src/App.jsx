import React from 'react';
import { RouterProvider, useRouter } from './context/RouterContext';
import Navbar from './components/layout/Navbar';
import Footer from './components/layout/Footer';

// 8 Feature Pages
import OverviewPage from './features/overview/OverviewPage';
import ChampionshipPage from './features/championship/ChampionshipPage';
import InsideTheScuderiaPage from './features/inside-the-scuderia/InsideTheScuderiaPage';
import RaceCalendarPage from './features/race-calendar/RaceCalendarPage';
import SimulationLabPage from './features/simulation-lab/SimulationLabPage';
import ModelIntelligencePage from './features/model-intelligence/ModelIntelligencePage';
import AiStrategistPage from './features/ai-strategist/AiStrategistPage';
import RetrospectivePage from './features/retrospective/RetrospectivePage';

function AppContent() {
  const { currentPath } = useRouter();

  const renderCurrentPage = () => {
    switch (currentPath) {
      case '/':
        return <OverviewPage />;
      case '/championship':
        return <ChampionshipPage />;
      case '/ferrari':
        return <InsideTheScuderiaPage />;
      case '/race-calendar':
        return <RaceCalendarPage />;
      case '/simulation-lab':
        return <SimulationLabPage />;
      case '/model-intelligence':
        return <ModelIntelligencePage />;
      case '/ai-strategist':
        return <AiStrategistPage />;
      case '/2026-retrospective':
        return <RetrospectivePage />;
      default:
        // Support hash variations or fallback to Overview
        if (currentPath.includes('championship')) return <ChampionshipPage />;
        if (currentPath.includes('ferrari')) return <InsideTheScuderiaPage />;
        if (currentPath.includes('calendar')) return <RaceCalendarPage />;
        if (currentPath.includes('simulation')) return <SimulationLabPage />;
        if (currentPath.includes('intelligence')) return <ModelIntelligencePage />;
        if (currentPath.includes('strategist') || currentPath.includes('ai')) return <AiStrategistPage />;
        if (currentPath.includes('retrospective') || currentPath.includes('2026')) return <RetrospectivePage />;
        return <OverviewPage />;
    }
  };

  return (
    <div className="min-h-screen bg-canvas-base text-text-primary flex flex-col font-body-md selection:bg-primary-container selection:text-text-primary">
      <Navbar />
      <main className="flex-1 w-full pt-16">
        {renderCurrentPage()}
      </main>
      <Footer />
    </div>
  );
}

export default function App() {
  return (
    <RouterProvider>
      <AppContent />
    </RouterProvider>
  );
}
