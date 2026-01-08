import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Dashboard from './components/Dashboard';
import PredictionView from './components/PredictionView';
import ChatBot from './components/ChatBot';
import './App.css';

const API_BASE_URL = 'http://localhost:5000/api';

function App() {
  const [historicalData, setHistoricalData] = useState([]);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('dashboard');

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const [histResponse, predResponse] = await Promise.all([
        axios.get(`${API_BASE_URL}/historical`),
        axios.get(`${API_BASE_URL}/predictions/2026`)
      ]);

      setHistoricalData(histResponse.data);
      setPredictions(predResponse.data);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching data:', error);
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="loading-screen">
        <h1>🏎️ Loading Ferrari Data...</h1>
      </div>
    );
  }

  return (
    <div className="App">
      <header className="app-header">
        <h1>🏎️ Ferrari F1 2026 Predictor</h1>
        <nav className="tabs">
          <button 
            className={activeTab === 'dashboard' ? 'active' : ''}
            onClick={() => setActiveTab('dashboard')}
          >
            📊 Dashboard
          </button>
          <button 
            className={activeTab === 'predictions' ? 'active' : ''}
            onClick={() => setActiveTab('predictions')}
          >
            🔮 Predictions
          </button>
          <button 
            className={activeTab === 'chat' ? 'active' : ''}
            onClick={() => setActiveTab('chat')}
          >
            💬 Chat
          </button>
        </nav>
      </header>

      <main className="app-content">
        {activeTab === 'dashboard' && (
          <Dashboard data={historicalData} />
        )}
        {activeTab === 'predictions' && (
          <PredictionView predictions={predictions} />
        )}
        {activeTab === 'chat' && (
          <ChatBot apiUrl={API_BASE_URL} />
        )}
      </main>
    </div>
  );
}

export default App;