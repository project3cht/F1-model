# main.py (with feature store integration)
"""
Main module for advanced F1 race predictions with feature store integration.
"""
import os
import pandas as pd
import numpy as np
import logging
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('f1_prediction')

# Import project modules
from features.feature_store_factory import FeatureStoreFactory
from config.config import load_config
from data.fetching import fetch_race_data, load_sample_data
from data.processing import calculate_driver_stats, calculate_team_stats
from models.predictor import AdvancedF1Predictor
from visualization.plots import plot_grid_vs_finish, plot_race_results
from utils.helpers import ensure_directory, save_figure

def run_prediction(track_name="Unknown Track", year=None, round_num=None, 
                  manual_data=None, safety_car_prob=0.6, rain_prob=0.0,
                  use_ml=True, trained_position_model=None, trained_interval_model=None,
                  config_path="config/config.yaml",
                  feature_store_config_path="config/feature_store_config.yaml"):
    """
    Run prediction for a specific race with advanced ML techniques.
    
    Args:
        track_name (str): Name of the track
        year (int): Year of the race
        round_num (int): Round number
        manual_data (dict): Manually provided data
        safety_car_prob (float): Probability of safety car
        rain_prob (float): Probability of rain
        use_ml (bool): Whether to use ML models or factor-based prediction
        trained_position_model (str): Path to trained position model
        trained_interval_model (str): Path to trained interval model
        config_path (str): Path to main configuration
        feature_store_config_path (str): Path to feature store configuration
        
    Returns:
        DataFrame: Race predictions
    """
    logger.info(f"Running prediction for {track_name}")
    
    # Step 1: Create feature store
    feature_store = FeatureStoreFactory.create_feature_store(feature_store_config_path)
    logger.info(f"Initialized feature store")
    
    # Step 2: Fetch qualifying data
    quali_data = fetch_race_data(year=year, round_num=round_num, manual_data=manual_data)
    logger.info(f"Fetched qualifying data with {len(quali_data)} drivers")
    
    # Add track information
    if track_name:
        quali_data['Track'] = track_name
    
    # Step 3: Prepare historical data for model training/context
    historical_data = load_sample_data(data_type='race')
    logger.info(f"Loaded historical data with {len(historical_data)} entries")
    
    # Step 4: Calculate driver and team stats
    driver_stats = calculate_driver_stats(historical_data)
    team_stats = calculate_team_stats(historical_data)
    logger.info(f"Calculated stats for {len(driver_stats)} drivers and {len(team_stats)} teams")
    
    # Step 5: Create predictor with feature store
    predictor = AdvancedF1Predictor()
    predictor.feature_store = feature_store  # Replace the default feature store
    
    # Step 6: Use ML or use pre-trained models if available
    if use_ml:
        if trained_position_model and trained_interval_model:
            # Load pre-trained models
            logger.info("Loading pre-trained models")
            predictor.load_models(trained_position_model, trained_interval_model)
        else:
            # Train models on historical data
            logger.info("Training new models on historical data")
            predictor.train(historical_data)
    
    # Step 7: Make predictions
    predictions = predictor.predict(
        quali_data, 
        driver_stats=driver_stats,
        team_stats=team_stats,
        safety_car_prob=safety_car_prob,
        rain_prob=rain_prob,
        track_name=track_name
    )
    logger.info("Generated race predictions")
    
    # Step 8: Create output directory
    results_dir = ensure_directory('results')
    
    # Step 9: Generate visualizations
    fig_grid_vs_finish = plot_grid_vs_finish(predictions, title=f"Grid vs Race Position - {track_name}")
    save_figure(fig_grid_vs_finish, f"{track_name}_grid_vs_position.png", results_dir)
    
    fig_race_results = plot_race_results(predictions, title=f"Race Results - {track_name}")
    save_figure(fig_race_results, f"{track_name}_race_results.png", results_dir)
    
    # Step 10: Save predictions to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(results_dir, f"{timestamp}_{track_name}_predictions.csv")
    predictions.to_csv(csv_filename, index=False)
    logger.info(f"Saved predictions to {csv_filename}")
    
    # Step 11: If new models were trained, save them
    if use_ml and not (trained_position_model and trained_interval_model):
        try:
            position_path, interval_path = predictor.save_models()
            logger.info(f"Saved new models to {position_path} and {interval_path}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    # Step 12: Clear feature store cache to save disk space
    feature_store.clear_cache()
    
    return predictions

def main():
    """Main function for command-line use."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Advanced F1 Race Prediction System')
    parser.add_argument('--track', type=str, default="Sample Track", help='Track name')
    parser.add_argument('--safety-car', type=float, default=0.6, help='Safety car probability (0-1)')
    parser.add_argument('--rain', type=float, default=0.0, help='Rain probability (0-1)')
    parser.add_argument('--use-ml', action='store_true', default=True, help='Use ML models for prediction')
    parser.add_argument('--position-model', type=str, default=None, help='Path to trained position model')
    parser.add_argument('--interval-model', type=str, default=None, help='Path to trained interval model')
    parser.add_argument('--config', type=str, default="config/config.yaml", help='Path to main configuration')
    parser.add_argument('--feature-store-config', type=str, default="config/feature_store_config.yaml", 
                       help='Path to feature store configuration')
    
    args = parser.parse_args()
    
    # Run prediction
    predictions = run_prediction(
        track_name=args.track,
        safety_car_prob=args.safety_car,
        rain_prob=args.rain,
        use_ml=args.use_ml,
        trained_position_model=args.position_model,
        trained_interval_model=args.interval_model,
        config_path=args.config,
        feature_store_config_path=args.feature_store_config
    )
    
    # Print top 10
    print("\nPredicted Top 10:")
    top10 = predictions.sort_values('Position').head(10)
    for _, row in top10.iterrows():
        print(f"{int(row['Position'])}. {row['Driver']} ({row['Team']}) - {row['Interval']}")

if __name__ == "__main__":
    main()