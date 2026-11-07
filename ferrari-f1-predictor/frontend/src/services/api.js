import axios from 'axios';
import frozenForecast2027 from '../data/preseason2027Data.json';

// Resolve API Base URL across Vercel Services binding (APP_URL), Vite env (VITE_API_BASE_URL), or relative public path (/api)
const resolveApiBaseUrl = () => {
  if (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL;
  }
  if (typeof process !== 'undefined' && process.env && process.env.APP_URL) {
    const url = process.env.APP_URL.replace(/\/$/, '');
    return url.endsWith('/api') ? url : `${url}/api`;
  }
  return '/api';
};

const API_BASE_URL = resolveApiBaseUrl();

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const get2027Predictions = async () => {
  try {
    const response = await apiClient.get('/predictions/2027');
    if (response.data && response.data.ferrari_constructors_forecast) {
      return response.data;
    }
  } catch (err) {
    console.warn('Backend /api/predictions/2027 not reachable, utilizing frozen preseason baseline artifact:', err.message);
  }
  return frozenForecast2027;
};

export const get2027Provenance = async () => {
  try {
    const response = await apiClient.get('/predictions/2027/provenance');
    if (response.data && response.data.code_commit_baseline) {
      return response.data;
    }
  } catch (err) {
    console.warn('Backend provenance endpoint not reachable, returning frozen metadata:', err.message);
  }
  return frozenForecast2027.freeze_metadata;
};

export const get2026Predictions = async () => {
  try {
    const response = await apiClient.get('/predictions/2026');
    return response.data;
  } catch (err) {
    console.warn('Backend /api/predictions/2026 not reachable:', err.message);
    return null;
  }
};

export const getHistoricalData = async (year) => {
  try {
    const url = year ? `/historical/${year}` : '/historical';
    const response = await apiClient.get(url);
    return response.data;
  } catch (err) {
    console.warn('Backend historical data not reachable:', err.message);
    return [];
  }
};

export const sendChatMessage = async (query) => {
  try {
    const response = await apiClient.post('/chat', { query });
    return response.data;
  } catch (err) {
    console.error('Chat API request error:', err);
    throw err;
  }
};

export default {
  get2027Predictions,
  get2027Provenance,
  get2026Predictions,
  getHistoricalData,
  sendChatMessage,
};

