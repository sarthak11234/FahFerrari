from fastf1.ergast import Ergast
import pandas as pd

ergast = Ergast()
try:
    print("Fetching 2024 full season...")
    response = ergast.get_race_results(season=2024, limit=1000)
    print("Content Type:", type(response.content))
    print("Content Length:", len(response.content))
    
    if len(response.content) > 0:
        first_item = response.content[0]
        print("First Item Type:", type(first_item))
        
        if isinstance(first_item, pd.DataFrame):
            print("First Item Columns:", first_item.columns.tolist())
            print("Head:\n", first_item.head(2))
        else:
            print("First Item:", first_item)
            
except Exception as e:
    print(f"Error: {e}")
