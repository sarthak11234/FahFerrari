import axios from 'axios';
import frozenForecast2027 from '../data/preseason2027Data.json';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000/api';

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

