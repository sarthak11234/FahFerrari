#!/bin/bash

# Test Ferrari F1 API endpoints

echo "🧪 Testing Ferrari F1 API"
echo "=========================="

# Health check
echo "\n1. Health Check:"
curl http://localhost:5000/api/health

# Historical data
echo "\n\n2. Historical Data (2025):"
curl http://localhost:5000/api/historical/2025

# Predictions
echo "\n\n3. 2026 Predictions:"
curl http://localhost:5000/api/predictions/2026

# Chat
echo "\n\n4. Chatbot Test:"
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How did Ferrari perform in 2025?"}'

# Summary stats
echo "\n\n5. Summary Statistics:"
curl http://localhost:5000/api/stats/summary

echo "\n\n✅ API tests complete!"