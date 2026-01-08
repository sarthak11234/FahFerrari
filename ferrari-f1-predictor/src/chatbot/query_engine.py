"""
Ferrari F1 Chatbot - Query Processing Engine
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import re

class FerrariChatbot:
    """Intelligent chatbot for Ferrari F1 queries"""
    
    def __init__(self, historical_data: pd.DataFrame, predictions: Dict):
        self.data = historical_data
        self.predictions = predictions
        self.conversation_history = []
        
        # Load latest season data
        self.latest_season = self.data[self.data['year'] == 2025].iloc[0]
    
    def parse_query(self, query: str) -> Dict:
        """
        Parse user query and extract intent
        
        Returns:
            Dictionary with query intent and entities
        """
        query_lower = query.lower()
        
        # Intent detection
        intents = {
            'season_results': ['2019', '2020', '2021', '2022', '2023', '2024', '2025'],
            'prediction': ['2026', 'predict', 'forecast', 'next season'],
            'driver': ['leclerc', 'hamilton', 'sainz', 'charles', 'lewis', 'carlos'],
            'comparison': ['compare', 'vs', 'versus', 'difference', 'better'],
            'technical': ['aero', 'engine', 'power unit', 'reliability', 'strategy'],
            'track': ['monaco', 'monza', 'spa', 'silverstone', 'singapore'],
            'statistics': ['stats', 'points', 'wins', 'podiums', 'pole', 'fastest lap']
        }
        
        detected_intents = []
        entities = []
        
        for intent, keywords in intents.items():
            for keyword in keywords:
                if keyword in query_lower:
                    detected_intents.append(intent)
                    entities.append(keyword)
        
        # Extract year if mentioned
        year_match = re.search(r'\b(19|20)\d{2}\b', query)
        year = int(year_match.group()) if year_match else None
        
        return {
            'intents': list(set(detected_intents)),
            'entities': list(set(entities)),
            'year': year,
            'original_query': query
        }
    
    def answer_season_query(self, year: int) -> str:
        """Answer questions about a specific season"""
        
        if year not in self.data['year'].values:
            return f"I don't have data for the {year} season yet."
        
        season_data = self.data[self.data['year'] == year].iloc[0]
        
        pos_val = season_data.get('position', 'N/A')
        try:
            pos_str = f"P{int(pos_val)}"
        except (ValueError, TypeError):
            pos_str = "N/A"
        
        response = f"""
📊 Ferrari in {year}:
- Championship Position: {pos_str}
- Total Points: {int(season_data.get('total_points', 0))}
- Wins: {int(season_data.get('wins', 0))}
- Podiums: {int(season_data.get('podiums', 0))}
- Average Race Position: P{season_data.get('avg_race_position', 0):.1f}
- Reliability Score: {int(season_data.get('reliability_score', 0))}/100
        """.strip()
        
        # Add context
        if year == 2025:
            response += "\n\n⚠️ 2025 was a tough year - first winless season since 2021. Hamilton struggled to adapt to the car."
        elif year == 2024:
            response += "\n\n✅ 2024 was Ferrari's highest points total ever (652)! Strong car but inconsistent strategy."
        
        return response
    
    def answer_prediction_query(self) -> str:
        """Answer questions about 2026 predictions"""
        
        pred = self.predictions
        
        # Get driver standing inputs if available
        drivers = pred.get('driver_standings', {})
        leclerc = drivers.get('leclerc', {'points': 0, 'rank_est': 'N/A'})
        hamilton = drivers.get('hamilton', {'points': 0, 'rank_est': 'N/A'})
        
        # Get constructors
        constructors = pred.get('constructors_standings', [])
        ferrari_const = next((c for c in constructors if c['team'] == 'Ferrari'), {'position': 'N/A'})
        
        response = f"""
🔮 Ferrari 2026 Detailed Forecast:

🏆 CHAMPIONSHIP OUTLOOK:
- Constructors: P{ferrari_const.get('position')} ({int(pred['predicted_points'])} pts)
- Drivers: Leclerc ({leclerc['rank_est']}), Hamilton ({hamilton['rank_est']})

🏎️ DRIVER BATTLE:
- Leclerc: {leclerc['points']} pts, {leclerc['wins']} wins
- Hamilton: {hamilton['points']} pts, {hamilton['wins']} wins

📅 RACE HIGHLIGHTS:
- Expected Wins: {pred['predicted_wins']} (Targeting Monaco, Monza, Singapore)
- Total Podiums: {pred['predicted_podiums']}

💡 STRATEGY INSIGHT:
- The model predicts a P{pred['predicted_position']} finish in the Constructors.
- To win, Ferrari must improve reliability by 15% and nail strategy calls.
        """.strip()
        
        if 'race_predictions' in pred:
            response += "\n\n🏁 NEXT 5 RACES (Simulated):\n"
            for race in pred['race_predictions'][:5]:
                response += f"- {race['circuit']}: {race['predicted_points']} pts ({race['win_probability']} win prob)\n"
                
        return response
    
    def answer_driver_query(self, driver: str) -> str:
        """Answer questions about specific drivers"""
        
        driver_lower = driver.lower()
        
        if 'leclerc' in driver_lower or 'charles' in driver_lower:
            return """
🏎️ Charles Leclerc (2025):
- Championship Position: P5
- Points: 242
- Podiums: 7
- Wins: 0 (tough season)
- Average Finish: P5.8

💪 Strengths:
- Exceptional qualifying pace
- Street circuit specialist (Monaco master)
- Strong racecraft and tire management
- Team leader since 2019

📊 Career at Ferrari (2019-2025):
- Total Wins: 8
- Total Podiums: 50+
- Pole Positions: 24
            """.strip()
        
        elif 'hamilton' in driver_lower or 'lewis' in driver_lower:
            return """
🏎️ Lewis Hamilton (2025 - First Ferrari Season):
- Championship Position: P6
- Points: 156
- Podiums: 0 (first time in career!)
- Wins: 0
- Average Finish: P7.2

⚠️ Difficult Adaptation:
- Struggled with Ferrari's driving style
- Car characteristics didn't suit his preferences
- Team strategy calls didn't click
- First season without a podium since 2007

🔮 2026 Outlook:
- Full year with the team should help
- Winter testing will be crucial
- Experience and 7 championships still valuable
- Needs car designed around both drivers
            """.strip()
        
        elif 'sainz' in driver_lower or 'carlos' in driver_lower:
            return """
🏎️ Carlos Sainz (moved to Williams in 2025):
- Final Ferrari season: 2024
- 2024 Stats: 296 points, 2 wins, 9 podiums
- Ferrari Career: 2021-2024
- Total Ferrari Wins: 3
- Known for consistency and racecraft

📍 Current: Racing for Williams in 2025
            """.strip()
        
        return "I can provide info about Leclerc or Hamilton (current Ferrari drivers) or Sainz (former driver)."
    
    def answer_comparison_query(self, entities: List[str]) -> str:
        """Compare different seasons or aspects"""
        
        years = [int(e) for e in entities if e.isdigit() and 2019 <= int(e) <= 2025]
        
        if len(years) >= 2:
            comparison = []
            for year in years:
                data = self.data[self.data['year'] == year].iloc[0]
                try:
                    pos = int(data.get('position', 0))
                except:
                    pos = "N/A"
                comparison.append(f"{year}: {int(data['total_points'])} pts, {int(data['wins'])} wins, P{pos}")
            
            return "📊 Season Comparison:\n" + "\n".join(comparison)
        
        # Compare 2024 vs 2025
        data_2024 = self.data[self.data['year'] == 2024].iloc[0]
        data_2025 = self.data[self.data['year'] == 2025].iloc[0]
        
        try:
             pos_2024 = str(int(data_2024.get('position', 0)))
        except:
             pos_2024 = "N/A"

        try:
             pos_2025 = str(int(data_2025.get('position', 0)))
        except:
             pos_2025 = "N/A"

        return f"""
📉 Ferrari: 2024 vs 2025

2024 (Strong Year):
- Points: {int(data_2024['total_points'])} | Position: P{pos_2024}
- Wins: {int(data_2024['wins'])} | Podiums: {int(data_2024['podiums'])}

2025 (Disaster Year):
- Points: {int(data_2025['total_points'])} | Position: P{pos_2025}
- Wins: {int(data_2025['wins'])} | Podiums: {int(data_2025['podiums'])}

DECLINE:
- -254 points (-39%)
- -5 wins
- -16 podiums
- Dropped from P3 to P4

🔴 What went wrong:
- Hamilton struggled to adapt
- Development direction missed the mark
- Strategy calls were poor
- Lost competitiveness mid-season
        """.strip()
    
    def answer_technical_query(self) -> str:
        """Answer technical questions"""
        
        latest = self.latest_season
        
        return f"""
🔧 Ferrari Technical Analysis (2025):

PERFORMANCE SCORES:
- Aerodynamics: {int(latest.get('aero_efficiency', 0))}/100
- Power Unit: {int(latest.get('power_unit', 0))}/100
- Reliability: {int(latest.get('reliability_score', 0))}/100
- Strategy: {int(latest.get('strategy_effectiveness', 0))}/100
- Development: {int(latest.get('development_pace', 0))}/100

STRENGTHS:
✅ Reliability improved vs 2024
✅ Power unit competitive
✅ Good in slow corners

WEAKNESSES:
❌ Aerodynamic efficiency dropped
❌ Lost high-speed performance
❌ Strategy calls cost points
❌ Development struggled to keep up

2026 EXPECTATIONS:
- Focus on aero recovery
- Better strategy tools
- New regulations could help
        """.strip()
    
    def generate_response(self, query: str) -> str:
        """
        Main function to generate chatbot response
        
        Args:
            query: User's question
            
        Returns:
            Chatbot response string
        """
        # Parse query
        parsed = self.parse_query(query)
        intents = parsed['intents']
        entities = parsed['entities']
        year = parsed['year']
        
        # Store in conversation history
        self.conversation_history.append({
            'query': query,
            'intents': intents,
            'timestamp': pd.Timestamp.now()
        })
        
        # Route to appropriate handler
        if 'prediction' in intents or '2026' in str(entities):
            return self.answer_prediction_query()
        
        elif 'season_results' in intents and year:
            return self.answer_season_query(year)
        
        elif 'driver' in intents:
            driver = next((e for e in entities if e in ['leclerc', 'hamilton', 'sainz', 'charles', 'lewis', 'carlos']), None)
            if driver:
                return self.answer_driver_query(driver)
        
        elif 'comparison' in intents:
            return self.answer_comparison_query(entities)
        
        elif 'technical' in intents:
            return self.answer_technical_query()
        
        else:
            # Default helpful response
            return """
🏎️ Ferrari F1 Chatbot - I can help with:

📊 SEASONS: Ask about any season 2019-2025
   • "How did Ferrari do in 2024?"
   • "Tell me about the 2025 season"

🔮 PREDICTIONS: 2026 forecasts
   • "What are the predictions for 2026?"
   • "Will Ferrari win in 2026?"

👨‍🔧 DRIVERS: Leclerc, Hamilton, Sainz
   • "Tell me about Charles Leclerc"
   • "How did Hamilton perform in 2025?"

📈 COMPARISONS: Season vs season
   • "Compare 2024 and 2025"

🔧 TECHNICAL: Performance analysis
   • "What are Ferrari's strengths?"
   • "Why did 2025 go wrong?"

Try asking a specific question!
            """.strip()

# CLI Interface
def run_chatbot_cli():
    """Run chatbot in command-line interface"""
    
    # Load data
    df = pd.read_csv('data/processed/ferrari_features.csv')
    
    # Load predictions (or generate them)
    predictions = {
        'predicted_points': 520,
        'confidence_interval': (480, 560),
        'predicted_position': 3,
        'predicted_wins': 4,
        'predicted_podiums': 18
    }
    
    chatbot = FerrariChatbot(df, predictions)
    
    print("\n" + "="*60)
    print("🏎️  FERRARI F1 CHATBOT")
    print("="*60)
    print("Ask me anything about Ferrari's F1 performance!")
    print("Type 'quit' to exit\n")
    
    while True:
        query = input("You: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nForza Ferrari! 🏎️🔴")
            break
        
        if not query:
            continue
        
        response = chatbot.generate_response(query)
        print(f"\nBot: {response}\n")

if __name__ == "__main__":
    run_chatbot_cli()