"""Ferrari F1 Data Collection Module
Fetches data from Ergast API and FastF1
"""

import requests
import pandas as pd
import fastf1
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime


class F1DataCollector:
    """Collect F1 data from various sources"""
    
    def __init__(self):
        self.ergast_base_url = "http://ergast.com/api/f1"
        fastf1.Cache.enable_cache('cache/')
    
    def fetch_season_results(self, year: int, team: str = "ferrari") -> pd.DataFrame:
        """
        Fetch season results from Ergast API
        
        Args:
            year: Season year
            team: Team constructor ID
            
        Returns:
            DataFrame with race results
        """
        url = f"{self.ergast_base_url}/{year}/constructors/{team}/results.json"
        response = requests.get(url)
        
        if response.status_code != 200:
            raise Exception(f"Failed to fetch data for {year}")
        
        data = response.json()
        races = data['MRData']['RaceTable']['Races']
        
        results = []
        for race in races:
            race_data = {
                'year': year,
                'round': race['round'],
                'race_name': race['raceName'],
                'circuit': race['Circuit']['circuitName'],
                'date': race['date']
            }
            
            for result in race['Results']:
                driver_result = race_data.copy()
                driver_result.update({
                    'driver': result['Driver']['familyName'],
                    'position': result['position'] if result['position'] != 'R' else None,
                    'points': float(result['points']),
                    'grid': int(result['grid']),
                    'status': result['status']
                })
                results.append(driver_result)
        
        return pd.DataFrame(results)
    
    def fetch_season_standings(self, year: int) -> Optional[Dict]:
        """Fetch constructor standings for a year"""
        url = f"{self.ergast_base_url}/{year}/constructorStandings.json"
        response = requests.get(url)
        
        data = response.json()
        standings = data['MRData']['StandingsTable']['StandingsLists'][0]['ConstructorStandings']
        
        for team in standings:
            if team['Constructor']['constructorId'] == 'ferrari':
                return {
                    'year': year,
                    'position': int(team['position']),
                    'points': float(team['points']),
                    'wins': int(team['wins'])
                }
        
        return None
    
    def collect_historical_data(self, start_year: int = 2019, end_year: int = 2025) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Collect all historical Ferrari data
        
        Args:
            start_year: Start year for data collection
            end_year: End year for data collection
            
        Returns:
            Complete historical dataset
        """
        all_results = []
        standings_data = []
        
        for year in range(start_year, end_year + 1):
            print(f"Fetching data for {year}...")
            
            # Get race results
            try:
                results = self.fetch_season_results(year)
                all_results.append(results)
            except Exception as e:
                print(f"Error fetching results for {year}: {e}")
            
            # Get standings
            try:
                standings = self.fetch_season_standings(year)
                if standings:
                    standings_data.append(standings)
            except Exception as e:
                print(f"Error fetching standings for {year}: {e}")
            
            # Rate limiting
            time.sleep(1)
        
        # Combine all data
        race_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
        standings_df = pd.DataFrame(standings_data)
        
        return race_df, standings_df
    
    def get_detailed_telemetry(self, year: int, race_round: int):
        """
        Fetch detailed telemetry using FastF1
        (Use sparingly - downloads large data)
        """
        session = fastf1.get_session(year, race_round, 'R')
        session.load()
        
        ferrari_drivers = session.drivers[:2]  # Top 2 drivers
        laps = session.laps.pick_drivers(ferrari_drivers)
        
        return laps


if __name__ == "__main__":
    collector = F1DataCollector()
    
    # Collect historical data
    race_data, standings_data = collector.collect_historical_data(2019, 2025)
    
    # Save to CSV
    race_data.to_csv('data/raw/race_results_2019_2025.csv', index=False)
    standings_data.to_csv('data/raw/standings_2019_2025.csv', index=False)
    
    print("✅ Data collection complete!")
