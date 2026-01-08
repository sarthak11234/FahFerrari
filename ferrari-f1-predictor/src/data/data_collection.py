"""
Ferrari F1 Data Collection
Migrated to use FastF1 (Ergast API) for reliable and fast data access
"""

import fastf1
from fastf1.ergast import Ergast
import pandas as pd
import os

class FerrariDataCollector:
    """Collect F1 data using FastF1 Ergast Interface"""
    
    def __init__(self):
        # Create cache directory
        cache_dir = 'cache'
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        fastf1.Cache.enable_cache(cache_dir)
        self.ergast = Ergast()
    
    def collect_historical_data(self, start_year: int = 2019, end_year: int = 2025) -> pd.DataFrame:
        """
        Collect all historical Ferrari data
        
        Args:
            start_year: Start year for data collection
            end_year: End year for data collection
            
        Returns:
            DataFrame of race results
        """
        all_results = []
        
        for year in range(start_year, end_year + 1):
            print(f"Fetching data for {year}...")
            
            try:
                # get_race_results returns a Result object
                # .content is a list of DataFrames (one per race)
                response = self.ergast.get_race_results(season=year, limit=1000)
                
                if not response.content:
                    print(f"  No data found for {year}")
                    continue
                    
                for race_df in response.content:
                    if race_df.empty:
                        continue
                        
                    # Columns usually include: season, round, url, raceName, date, time, 
                    # driverId, givenName, familyName, dateOfBirth, nationality, 
                    # constructorId, constructorName, constructorNationality, ...
                    # number, position, positionText, points, wins, grid, laps, status, ...
                    
                    # Filter for Ferrari
                    # Note: constructorId for Ferrari is 'ferrari'
                    if 'constructorId' not in race_df.columns:
                        print(f"  Warning: No constructorId column in round {race_df.iloc[0].get('round', 'unknown')}")
                        continue
                        
                    ferrari_results = race_df[race_df['constructorId'] == 'ferrari']
                    
                    for _, row in ferrari_results.iterrows():
                        all_results.append({
                            'year': int(row.get('season', year)),
                            'round': int(row.get('round', 0)),
                            'points': float(row.get('points', 0)),
                            'position': str(row.get('position', 'D')),
                            'grid': int(row.get('grid', 0)),
                            'status': str(row.get('status', 'Finished'))
                        })
                            
            except Exception as e:
                print(f"Error fetching data for {year}: {e}")
                
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        if not df.empty:
            print(f"✅ Collected {len(df)} race results!")
        else:
            print("⚠️ No results collected!")
            
        return df

# Usage
if __name__ == "__main__":
    collector = FerrariDataCollector()
    
    # Collect historical data
    race_data = collector.collect_historical_data(2019, 2025)
    
    # Save to CSV
    os.makedirs('data/raw', exist_ok=True)
    if not race_data.empty:
        race_data.to_csv('data/raw/race_results_2019_2025.csv', index=False)
        print("✅ Data saved to CSV!")
    else:
        print("❌ No data to save.")
