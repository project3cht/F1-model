# main.py (fixed)
import os
import pandas as pd
import numpy as np
import logging
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

# Create __init__.py files to allow proper imports
def ensure_init_files():
    """Create __init__.py files in all directories to ensure proper imports."""
    dirs = [
        "domain", "application", "infrastructure", "infrastructure/persistence", 
        "infrastructure/external_services", "application/dto", "models", 
        "features", "data", "visualization", "utils", "config"
    ]
    
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        init_file = os.path.join(directory, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write("# Auto-generated __init__.py file\n")

# Call this at the beginning to ensure imports work
ensure_init_files()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('f1_prediction')

# Import project modules
try:
    from features.feature_store_factory import FeatureStoreFactory
    from config.config import load_config
    from data.fetching import fetch_race_data, load_sample_data
    from data.processing import calculate_driver_stats, calculate_team_stats
    from models.predictor_enhanced import EnhancedF1Predictor
    from visualization.base import plot_grid_vs_finish, plot_race_results, save_figure
    from utils.helpers import ensure_directory
    from utils.constants import TRACK_CHARACTERISTICS, DRIVERS, TEAMS
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure all required modules are properly installed")
    raise

# Import visualization modules - these are optional, don't fail if they're not available
try:
    from visualization.race import plot_grid_vs_finish, plot_top10_intervals, plot_position_changes
    from visualization.team import plot_team_performance, plot_team_pace_comparison
    visualization_modules_available = True
except ImportError:
    logger.warning("Visualization modules not fully available")
    visualization_modules_available = False

# Function to find a track by name in our track characteristics data
def find_track_by_name(track_name):
    """Find the closest matching track in the track characteristics."""
    if track_name in TRACK_CHARACTERISTICS:
        return track_name
    
    # Try a fuzzy match
    for known_track in TRACK_CHARACTERISTICS.keys():
        if track_name.lower() in known_track.lower() or known_track.lower() in track_name.lower():
            return known_track
    
    # Try matching with common shortened names
    track_aliases = {
        "Melbourne": "Albert Park",
        "Spielberg": "Red Bull Ring",
        "Budapest": "Hungaroring",
        "Monza": "Monza",
        "Miami": "Miami",
        "COTA": "Austin",
        "Vegas": "Las Vegas",
        "Abu Dhabi": "Abu Dhabi",
        "Albert Park": "Melbourne",
        "Shanghai": "Shanghai",
        "Jeddah": "Jeddah",
        "Imola": "Imola",
        "Monaco": "Monaco",
        "Barcelona": "Barcelona",
        "Montreal": "Montreal",
        "Silverstone": "Silverstone"
    }
    
    for alias, full_name in track_aliases.items():
        if alias.lower() in track_name.lower():
            for known_track in TRACK_CHARACTERISTICS.keys():
                if full_name.lower() in known_track.lower():
                    return known_track
    
    # No match found
    return None

# Historical safety car and rain probability model - simplified for our fix
class TrackConditionPredictor:
    """Predict safety car and rain probabilities based on historical data."""
    
    def __init__(self):
        self.safety_car_history = {}
        self.rain_history = {}
        self.track_characteristics = TRACK_CHARACTERISTICS
        
    def load_historical_data(self):
        """Load historical data for safety car deployments and rain conditions."""
        logger.info("Loading historical track condition data")
        # This is a simplified implementation
        pass
    
    def predict_safety_car_probability(self, track_name):
        """Predict safety car probability for a given track."""
        if track_name in self.track_characteristics:
            layout_stress = self.track_characteristics[track_name]['layout_stress']
            return 0.4 + (layout_stress * 0.5)  # Scale between 0.4 and 0.9
        return 0.6  # Default if no data
    
    def predict_rain_probability(self, track_name):
        """Predict rain probability for a given track."""
        if track_name in self.track_characteristics:
            temp = self.track_characteristics[track_name]['temperature']
            return max(0, min(0.6, (30 - temp) / 40))
        return 0.2  # Default if no data

def get_race_info(round_num, year=2025):
    """Get simplified race information for a round."""
    # Simple mapping of rounds to tracks for demonstration
    tracks = {
        1: "Bahrain",
        2: "Jeddah",
        3: "Melbourne",
        4: "Suzuka",
        5: "Shanghai",
        6: "Miami",
        7: "Imola",
        8: "Monaco",
        9: "Montreal",
        10: "Barcelona",
        11: "Spielberg",
        12: "Silverstone",
        13: "Budapest",
        14: "Spa",
        15: "Zandvoort",
        16: "Monza",
        17: "Baku",
        18: "Singapore",
        19: "Austin",
        20: "Mexico City",
        21: "Sao Paulo",
        22: "Las Vegas",
        23: "Qatar",
        24: "Abu Dhabi"
    }
    
    if round_num not in tracks:
        return {
            "track": f"Unknown Track (Round {round_num})",
            "name": f"Race {round_num}",
            "safety_car_prob": 0.6,
            "rain_prob": 0.2
        }
    
    track_name = tracks[round_num]
    race_name = f"{track_name} Grand Prix"
    
    # Initialize track condition predictor
    predictor = TrackConditionPredictor()
    predictor.load_historical_data()
    
    # Get known track from our database
    known_track = find_track_by_name(track_name)
    
    # Predict safety car and rain probabilities
    if known_track:
        safety_car_prob = predictor.predict_safety_car_probability(known_track)
        rain_prob = predictor.predict_rain_probability(known_track)
    else:
        # Default values if we can't match the track
        safety_car_prob = 0.6
        rain_prob = 0.2
    
    return {
        "track": track_name,
        "name": race_name,
        "safety_car_prob": safety_car_prob,
        "rain_prob": rain_prob
    }

def run_prediction_for_round(round_num, use_monte_carlo=True, config_path="config/config.yaml", year=2025):
    """
    Run prediction for a specific F1 round with enhanced ML techniques and visualizations.
    
    Args:
        round_num (int): F1 race round number
        use_monte_carlo (bool): Whether to use Monte Carlo simulation
        config_path (str): Path to configuration file
        year (int): F1 season year
        
    Returns:
        DataFrame: Race predictions
    """
    # Get race info
    race_info = get_race_info(round_num, year)
    
    track_name = race_info["track"]
    race_name = race_info["name"]
    safety_car_prob = race_info["safety_car_prob"]
    rain_prob = race_info["rain_prob"]
    
    logger.info(f"Running prediction for Round {round_num}: {race_name} ({track_name})")
    logger.info(f"Safety Car Probability: {safety_car_prob:.2f}, Rain Probability: {rain_prob:.2f}")
    
    # Load configuration
    try:
        config = load_config(config_path)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.warning(f"Error loading config: {e}")
        logger.info("Using default configuration")
        from dataclasses import dataclass
        
        @dataclass
        class SimpleConfig:
            class model:
                model_type = 'ensemble'
                params = {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05}
                features = ['GridPosition', 'TeamPerformanceFactor', 'DriverPerformanceFactor']
            
            class data:
                historical_data_path = 'data/historical_races.csv'
                sample_size = 1000
                validation_split = 0.2
            
            class prediction:
                default_safety_car_prob = 0.6
                default_rain_prob = 0.0
                simulation_runs = 1000
                
            log_level = 'INFO'
            output_dir = 'results'
        
        config = SimpleConfig()
    
    # Create feature store
    try:
        feature_store = FeatureStoreFactory.create_feature_store(config_path)
        logger.info("Feature store initialized")
    except Exception as e:
        logger.warning(f"Error creating feature store: {e}")
        from features.feature_store import FeatureStore
        feature_store = FeatureStore()
        logger.info("Created fallback feature store")
    
    # Fetch qualifying data
    try:
        quali_data = fetch_race_data(year=year, round_num=round_num)
        if quali_data is not None and len(quali_data) > 0:
            logger.info(f"Fetched qualifying data with {len(quali_data)} drivers")
        else:
            raise ValueError("No qualifying data returned")
    except Exception as e:
        logger.warning(f"Could not get qualifying data: {e}")
        logger.info("Using sample qualifying data with current drivers")
        quali_data = load_sample_data(data_type='qualifying')
    
    # Add track information
    if track_name:
        quali_data['Track'] = track_name
    
    # Load/prepare historical data
    try:
        historical_data = load_sample_data(data_type='race')
        logger.info(f"Loaded historical data with {len(historical_data)} entries")
    except Exception as e:
        logger.warning(f"Error loading historical data: {e}")
        historical_data = pd.DataFrame()
    
    # Calculate driver and team stats
    try:
        driver_stats = calculate_driver_stats(historical_data)
        team_stats = calculate_team_stats(historical_data)
        logger.info(f"Calculated stats for {len(driver_stats)} drivers and {len(team_stats)} teams")
    except Exception as e:
        logger.warning(f"Error calculating stats: {e}")
        driver_stats = None
        team_stats = None
    
    # Create and configure predictor
    try:
        predictor = EnhancedF1Predictor(config_path=config_path)
        logger.info("Created EnhancedF1Predictor")
    except Exception as e:
        logger.error(f"Error creating predictor: {e}")
        from models.predictor import AdvancedF1Predictor
        predictor = AdvancedF1Predictor(name="FallbackPredictor")
        logger.info("Created fallback AdvancedF1Predictor")
    
    # Try to load pre-trained models
    model_loaded = False
    try:
        model_dir = "models/saved"
        predictor.load_models(model_dir)
        logger.info(f"Loaded pre-trained models from {model_dir}")
        model_loaded = True
    except Exception as e:
        logger.warning(f"Could not load pre-trained models: {e}")
    
    # Train models if needed
    if not model_loaded:
        try:
            logger.info("Training new models on historical data")
            predictor.train(historical_data)
        except Exception as e:
            logger.error(f"Error training models: {e}")
            # Create a very simple DataFrame with predictions based on grid positions
            predictions = quali_data.copy()
            predictions['Position'] = predictions['Grid'].values
            predictions['Interval'] = ["WINNER" if i == 0 else f"+{i*1.5:.3f}s" for i in range(len(predictions))]
            predictions['Interval (s)'] = [0 if i == 0 else i*1.5 for i in range(len(predictions))]
            predictions['Points'] = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + [0] * (len(predictions) - 10)
            return predictions
    
    # Make predictions
    try:
        logger.info(f"Making predictions with safety_car_prob={safety_car_prob}, rain_prob={rain_prob}")
        
        if use_monte_carlo:
            # Run Monte Carlo simulation for more robust predictions with uncertainty
            try:
                predictions = predictor.predict(
                    quali_data=quali_data,
                    driver_stats=driver_stats,
                    team_stats=team_stats,
                    safety_car_prob=safety_car_prob,
                    rain_prob=rain_prob,
                    use_monte_carlo=True,
                    return_uncertainty=True
                )
                
                # Extract main predictions
                if isinstance(predictions, dict):
                    prediction_df = predictions['predictions']
                    logger.info("Generated race predictions with uncertainty using Monte Carlo simulation")
                else:
                    prediction_df = predictions
                    logger.info("Generated race predictions")
            except Exception as e:
                logger.warning(f"Error in Monte Carlo simulation: {e}")
                predictions = predictor.predict(
                    quali_data=quali_data,
                    driver_stats=driver_stats,
                    team_stats=team_stats,
                    safety_car_prob=safety_car_prob,
                    rain_prob=rain_prob,
                    use_monte_carlo=False
                )
                prediction_df = predictions
                logger.info("Generated race predictions without Monte Carlo")
        else:
            # Standard prediction
            predictions = predictor.predict(
                quali_data=quali_data,
                driver_stats=driver_stats,
                team_stats=team_stats,
                safety_car_prob=safety_car_prob,
                rain_prob=rain_prob
            )
            prediction_df = predictions
            logger.info("Generated race predictions")
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        # Create a simple fallback prediction
        prediction_df = quali_data.copy()
        prediction_df['Position'] = prediction_df['Grid'].values
        prediction_df['Interval'] = ["WINNER" if i == 0 else f"+{i*1.5:.3f}s" for i in range(len(prediction_df))]
        prediction_df['Interval (s)'] = [0 if i == 0 else i*1.5 for i in range(len(prediction_df))]
        prediction_df['Points'] = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + [0] * (len(prediction_df) - 10)
    
    # Create output directory
    results_dir = ensure_directory('results')
    race_dir = ensure_directory(os.path.join(results_dir, f"round_{round_num}_{race_name.replace(' ', '_')}"))
    viz_dir = ensure_directory(os.path.join(race_dir, "visualizations"))
    
    # Create and save basic visualizations
    try:
        # Grid vs Finish
        fig_grid = plot_grid_vs_finish(
            prediction_df, 
            title=f"Grid vs Race Position - {race_name}",
            save_path=os.path.join(race_dir, "grid_vs_finish.png")
        )
        plt.close(fig_grid)
        
        # Create more visualizations if the modules are available
        if visualization_modules_available:
            try:
                # Top 10 Intervals
                plot_top10_intervals(prediction_df, folder_path=race_dir)
                
                # Position Changes
                plot_position_changes(prediction_df, folder_path=race_dir)
                
                # Team Performance
                plot_team_performance(prediction_df, save_path=os.path.join(race_dir, "team_performance.png"))
                
                # Team Pace Comparison
                plot_team_pace_comparison(prediction_df, folder_path=race_dir)
                
                logger.info("Created visualizations")
            except Exception as e:
                logger.warning(f"Error creating additional visualizations: {e}")
        
    except Exception as e:
        logger.warning(f"Error creating visualizations: {e}")
    
    # Save predictions to CSV
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join(race_dir, f"{timestamp}_predictions.csv")
        prediction_df.to_csv(csv_filename, index=False)
        logger.info(f"Saved predictions to {csv_filename}")
    except Exception as e:
        logger.warning(f"Error saving predictions to CSV: {e}")
    
    # Format text output for console display
    try:
        # Create a simple formatted output
        formatted_output = "\n🏁 Predicted Race Results 🏁\n"
        formatted_output += "=" * 70 + "\n"
        formatted_output += f"{'Pos':<5}{'Driver':<20}{'Team':<25}{'Grid':<8}{'Interval':<15}\n"
        formatted_output += "-" * 70 + "\n"
        
        for idx, row in prediction_df.head(10).iterrows():
            formatted_output += f"{int(row['Position']):<5}{row['Driver']:<20}{row['Team']:<25}{int(row['Grid']):<8}{row['Interval']:<15}\n"
        
        # Print race info
        print(f"\nRace Information for Round {round_num}:")
        print(f"Track: {track_name}")
        print(f"Race: {race_name}")
        print(f"Safety Car Probability: {safety_car_prob:.2f}")
        print(f"Rain Probability: {rain_prob:.2f}")
        
        # Print top 10 to console
        print(formatted_output)
        
    except Exception as e:
        logger.warning(f"Error formatting output: {e}")
        print("Prediction completed but could not format output.")
    
    return prediction_df

def main():
    """Main function for command-line use."""
    parser = argparse.ArgumentParser(description='F1 Race Prediction System')
    parser.add_argument('--round', type=int, required=True, help='F1 race round number')
    parser.add_argument('--no-monte-carlo', action='store_true', help='Disable Monte Carlo simulation')
    parser.add_argument('--config', type=str, default="config/config.yaml", help='Path to configuration file')
    parser.add_argument('--year', type=int, default=2025, help='F1 season year')
    
    args = parser.parse_args()
    
    # Run prediction for specified round
    run_prediction_for_round(
        round_num=args.round,
        use_monte_carlo=not args.no_monte_carlo,
        config_path=args.config,
        year=args.year
    )

if __name__ == "__main__":
    main()