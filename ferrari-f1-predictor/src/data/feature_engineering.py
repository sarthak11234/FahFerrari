"""
Ferrari F1 Feature Engineering
Creates advanced features for ML models
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, List

class FerrariFeatureEngineer:
    """Create advanced features from raw F1 data"""
    
    def __init__(self):
        self.feature_list = []
    
    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic aggregated features"""
        print(f"DEBUG: df columns: {df.columns.tolist()}")
        
        # Create a numeric position column for aggregation
        # 'D', 'R', etc. will become NaN and excluded from mean calculation
        df['position_num'] = pd.to_numeric(df['position'], errors='coerce')
        
        # Group by year
        yearly = df.groupby('year').agg({
            'points': ['sum', 'mean', 'std'],
            'position_num': 'mean',  # Use numeric position
            'grid': 'mean'
        }).reset_index()
        
        yearly.columns = ['year', 'total_points', 'avg_points_per_race', 
                         'points_std', 'avg_race_position', 'avg_grid']
        
        # Count wins and podiums based on original string position
        # We need to ensure we are matching the correct strings '1', '2', '3'
        yearly['wins'] = df[df['position'].astype(str) == '1'].groupby('year').size().reindex(yearly['year']).fillna(0).values
        yearly['podiums'] = df[df['position'].astype(str).isin(['1','2','3'])].groupby('year').size().reindex(yearly['year']).fillna(0).values
        
        # DNFs are usually marked in Status or if position is not numeric key (like 'R', 'D')
        # But in data_collector we saved status.
        yearly['dnfs'] = df[df['status'] != 'Finished'].groupby('year').size().reindex(yearly['year']).fillna(0).values
        
        yearly = yearly.fillna(0)
        
        return yearly
    
    def create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trend and momentum features"""
        
        df = df.sort_values('year')
        
        # Points trends
        df['points_diff'] = df['total_points'].diff()
        df['points_pct_change'] = df['total_points'].pct_change()
        
        # Moving averages (2-year window)
        df['points_ma2'] = df['total_points'].rolling(window=2, min_periods=1).mean()
        df['wins_ma2'] = df['wins'].rolling(window=2, min_periods=1).mean()
        
        # Momentum indicators
        df['improving'] = (df['points_diff'] > 0).astype(int)
        df['consecutive_improvements'] = (df['improving'] * 
                                          (df['improving'].groupby(
                                              (df['improving'] != df['improving'].shift()).cumsum()
                                          ).cumcount() + 1))
        
        return df
    
    def create_performance_indices(self, df: pd.DataFrame, tech_df: pd.DataFrame) -> pd.DataFrame:
        """Create composite performance indices"""
        
        # Merge technical scores
        df = df.merge(tech_df, on='year', how='left')
        
        # Race pace index
        df['race_pace_index'] = (
            (25 - df['avg_race_position']) * 10 +
            df['avg_points_per_race'] * 2 +
            df['wins'] * 15
        )
        
        # Car competitiveness index
        df['car_competitiveness'] = (
            df['aero_efficiency'] * 0.35 +
            df['power_unit'] * 0.30 +
            df['reliability_score'] * 0.35
        )
        
        # Team performance index
        df['team_performance'] = (
            df['strategy_effectiveness'] * 0.25 +
            df['development_pace'] * 0.25 +
            df['driver_skill'] * 0.30 +
            df['car_competitiveness'] * 0.20
        )
        
        # Efficiency metrics
        df['points_per_race'] = df['total_points'] / 24  # Assuming 24 races
        df['podium_rate'] = df['podiums'] / 48  # 2 drivers * 24 races
        df['dnf_rate'] = df['dnfs'] / 48
        df['win_rate'] = df['wins'] / 24
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2]) -> pd.DataFrame:
        """Create lagged features for time series"""
        
        df = df.sort_values('year')
        
        for lag in lags:
            df[f'points_lag{lag}'] = df['total_points'].shift(lag)
            df[f'position_lag{lag}'] = df['avg_race_position'].shift(lag)
            df[f'wins_lag{lag}'] = df['wins'].shift(lag)
        
        return df
    
    def engineer_all_features(self, race_df: pd.DataFrame, 
                             tech_df: pd.DataFrame) -> pd.DataFrame:
        """Complete feature engineering pipeline"""
        
        print("🔧 Creating basic features...")
        df = self.create_basic_features(race_df)
        
        print("📈 Creating trend features...")
        df = self.create_trend_features(df)
        
        print("🎯 Creating performance indices...")
        df = self.create_performance_indices(df, tech_df)
        
        print("⏱️ Creating lag features...")
        df = self.create_lag_features(df)
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(0)
        
        # Store feature names
        self.feature_list = df.columns.tolist()
        
        print(f"✅ Created {len(df.columns)} features!")
        
        return df

# Usage
if __name__ == "__main__":
    # Load data
    race_df = pd.read_csv('data/raw/race_results_2019_2025.csv')
    print(f"DEBUG: Loaded columns: {race_df.columns.tolist()}")
    print(f"DEBUG: Sample data:\n{race_df.head()}")
    tech_df = pd.read_csv('data/raw/technical_scores.csv')
    
    # Engineer features
    engineer = FerrariFeatureEngineer()
    features_df = engineer.engineer_all_features(race_df, tech_df)
    
    # Save
    os.makedirs('data/processed', exist_ok=True)
    features_df.to_csv('data/processed/ferrari_features.csv', index=False)
    print("✅ Features saved!")