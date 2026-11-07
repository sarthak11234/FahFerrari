#!/bin/bash

# Test Ferrari F1 API endpoints
API_HOST="${APP_URL:-http://localhost:5000}"
API_HOST="${API_HOST%/}"

echo "🧪 Testing Ferrari F1 API on ${API_HOST}"
echo "=========================="

# Health check
echo -e "\n1. Health Check:"
curl "${API_HOST}/api/health"

# Historical data
echo -e "\n\n2. Historical Data (2025):"
curl "${API_HOST}/api/historical/2025"

# Predictions
echo -e "\n\n3. 2026 Predictions:"
curl "${API_HOST}/api/predictions/2026"

# Chat
echo -e "\n\n4. Chatbot Test:"
curl -X POST "${API_HOST}/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "How did Ferrari perform in 2025?"}'

# Summary stats
echo -e "\n\n5. Summary Statistics:"
curl "${API_HOST}/api/stats/summary"

echo -e "\n\n✅ API tests complete!"