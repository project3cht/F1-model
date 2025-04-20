# main.py (enhanced with FastF1 schedule and probability predictions)
import os
import pandas as pd
import numpy as np
import logging
import argparse
from datetime import datetime
import fastf1

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
from models.predictor_enhanced import EnhancedF1Predictor
from visualization.plots import plot_grid_vs_finish, plot_race_results
from utils.helpers import ensure_directory, save_figure

# Constants from utils/constants.py
from utils.constants import TRACK_CHARACTERISTICS, DRIVERS, TEAMS

# Load FastF1 schedule for current season
def load_f1_schedule(year=2025):
    """Load F1 schedule for the current year using FastF1."""
    try:
        schedule = fastf1.get_event_schedule(year)
        return schedule
    except Exception as e:
        logger.warning(f"Could not load FastF1 schedule: {e}")
        logger.info("Falling back to basic schedule")
        return None

# Historical safety car and rain probability model
class TrackConditionPredictor:
    """Predict safety car and rain probabilities based on historical data."""
    
    def __init__(self):
        self.safety_car_history = {}
        self.rain_history = {}
        self.track_characteristics = TRACK_CHARACTERISTICS
        
    def load_historical_data(self, years_range=[2020, 2021, 2022, 2023, 2024]):
        """Load historical data for safety car deployments and rain conditions."""
        logger.info("Loading historical track condition data")
        
        # This is where you would ideally load from a database or API
        # For now, we'll use some basic historical statistics
        # In a real application, you could use fastf1 to get historical data
        
        try:
            for year in years_range:
                try:
                    schedule = fastf1.get_event_schedule(year)
                    for _, event in schedule.iterrows():
                        try:
                            track_name = event['EventName']
                            # Here you would load the race session and check if safety car was deployed
                            # and if it was raining during the race
                            
                            # For demonstration purposes, we'll use the track characteristics
                            if track_name not in self.safety_car_history:
                                self.safety_car_history[track_name] = []
                                self.rain_history[track_name] = []
                            
                            # Try to get actual race data from FastF1
                            try:
                                race = fastf1.get_session(year, event['RoundNumber'], 'R')
                                race.load()
                                
                                # Check for safety car (a more sophisticated approach would be needed)
                                # This is just an example
                                if hasattr(race, 'laps') and hasattr(race.laps, 'Status'):
                                    sc_deployed = any(race.laps['Status'].str.contains('SAFETY', na=False))
                                    self.safety_car_history[track_name].append(1 if sc_deployed else 0)
                                
                                # Check for rain (more sophisticated detection would be needed)
                                # For simplicity, let's just use track characteristics abrasiveness for now
                                if track_name in self.track_characteristics:
                                    self.rain_history[track_name].append(
                                        1 if np.random.random() < self.track_characteristics[track_name]['abrasiveness'] / 2 else 0
                                    )
                            except Exception as e:
                                # If we can't load actual data, use approximations
                                if track_name in self.track_characteristics:
                                    layout_stress = self.track_characteristics[track_name]['layout_stress']
                                    self.safety_car_history[track_name].append(
                                        1 if np.random.random() < layout_stress else 0
                                    )
                                    
                                    # For rain, use a random value biased by track temperature
                                    temp = self.track_characteristics[track_name]['temperature']
                                    rain_chance = max(0, min(1, (30 - temp) / 30))
                                    self.rain_history[track_name].append(
                                        1 if np.random.random() < rain_chance else 0
                                    )
                        except Exception as inner_e:
                            logger.warning(f"Could not process {year} race at {track_name}: {inner_e}")
                            
                except Exception as schedule_e:
                    logger.warning(f"Could not load {year} schedule: {schedule_e}")
            
            logger.info(f"Loaded historical data for {len(self.safety_car_history)} tracks")
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    def predict_safety_car_probability(self, track_name):
        """Predict safety car probability for a given track."""
        if track_name in self.safety_car_history and len(self.safety_car_history[track_name]) > 0:
            # Calculate probability from historical data
            prob = sum(self.safety_car_history[track_name]) / len(self.safety_car_history[track_name])
            
            # Adjust with track characteristics if available
            if track_name in self.track_characteristics:
                layout_stress = self.track_characteristics[track_name]['layout_stress']
                abrasiveness = self.track_characteristics[track_name]['abrasiveness']
                
                # Weight historical data (70%) and track characteristics (30%)
                prob = 0.7 * prob + 0.3 * ((layout_stress + abrasiveness) / 2)
            
            return max(0.3, min(0.9, prob))  # Ensure probability is between 0.3 and 0.9
        
        # Fallback to default probabilities based on track type
        if track_name in self.track_characteristics:
            layout_stress = self.track_characteristics[track_name]['layout_stress']
            return 0.4 + (layout_stress * 0.5)  # Scale between 0.4 and 0.9
        
        # Default if no data
        return 0.6
    
    def predict_rain_probability(self, track_name):
        """Predict rain probability for a given track."""
        if track_name in self.rain_history and len(self.rain_history[track_name]) > 0:
            # Calculate probability from historical data
            prob = sum(self.rain_history[track_name]) / len(self.rain_history[track_name])
            
            # Adjust with track characteristics if available
            if track_name in self.track_characteristics:
                temp = self.track_characteristics[track_name]['temperature']
                rain_prob = max(0, min(0.6, (30 - temp) / 40))
                
                # Weight historical data (60%) and temperature-based probability (40%)
                prob = 0.6 * prob + 0.4 * rain_prob
            
            return max(0.0, min(0.7, prob))  # Ensure probability is between 0 and 0.7
        
        # Fallback to default probabilities based on track temperature
        if track_name in self.track_characteristics:
            temp = self.track_characteristics[track_name]['temperature']
            return max(0, min(0.6, (30 - temp) / 40))
        
        # Default if no data
        return 0.2

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

def get_race_info_from_schedule(round_num, year=2025):
    """Get race information from FastF1 schedule."""
    try:
        schedule = load_f1_schedule(year)
        if schedule is None:
            return None
        
        # Find the event with matching round number
        event = schedule.get_event_by_round(round_num)
        if event is None:
            return None
        
        # Extract track name and race name
        track_name = event["Location"]
        race_name = event["EventName"]
        
        # Find matching track in our track characteristics
        known_track = find_track_by_name(track_name)
        
        # Initialize track condition predictor
        predictor = TrackConditionPredictor()
        predictor.load_historical_data()
        
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
    except Exception as e:
        logger.error(f"Error getting race info from FastF1 schedule: {e}")
        return None

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
    # Get race info from FastF1 schedule
    race_info = get_race_info_from_schedule(round_num, year)
    
    # If we couldn't get race info from FastF1, use fallback values
    if race_info is None:
        logger.warning(f"Could not get race info for round {round_num} from FastF1 schedule")
        logger.info("Using fallback values")
        race_info = {
            "track": f"Track for Round {round_num}",
            "name": f"Race {round_num}",
            "safety_car_prob": 0.6,
            "rain_prob": 0.2
        }
    
    track_name = race_info["track"]
    race_name = race_info["name"]
    safety_car_prob = race_info["safety_car_prob"]
    rain_prob = race_info["rain_prob"]
    
    logger.info(f"Running prediction for Round {round_num}: {race_name} ({track_name})")
    logger.info(f"Safety Car Probability: {safety_car_prob:.2f}, Rain Probability: {rain_prob:.2f}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Create feature store
    feature_store = FeatureStoreFactory.create_feature_store(config_path)
    logger.info(f"Initialized feature store")
    
    # Fetch qualifying data
    quali_data = fetch_race_data(year=year, round_num=round_num)
    logger.info(f"Fetched qualifying data with {len(quali_data)} drivers")
    
    # If we couldn't get qualifying data, use sample data with current drivers
    if quali_data is None or len(quali_data) == 0:
        logger.warning(f"Could not get qualifying data for round {round_num}")
        logger.info("Using sample qualifying data with current drivers")
        
        # Generate synthetic qualifying data using drivers from constants.py
        drivers_data = []
        for i, (driver, team) in enumerate(DRIVERS.items()):
            drivers_data.append({
                'Driver': driver,
                'Team': team,
                'GridPosition': i + 1
            })
        
        quali_data = pd.DataFrame(drivers_data)
    
    # Add track information
    if track_name:
        quali_data['Track'] = track_name
    
    # Prepare historical data for context
    historical_data = load_sample_data(data_type='race')
    logger.info(f"Loaded historical data with {len(historical_data)} entries")
    
    # Calculate driver and team stats
    driver_stats = calculate_driver_stats(historical_data)
    team_stats = calculate_team_stats(historical_data)
    logger.info(f"Calculated stats for {len(driver_stats)} drivers and {len(team_stats)} teams")
    
    # Create enhanced predictor
    predictor = EnhancedF1Predictor(config_path=config_path)
    
    # Try to load pre-trained models from the models directory
    try:
        model_dir = "models"
        if hasattr(config, 'model') and hasattr(config.model, 'model_dir'):
            model_dir = config.model.model_dir
        
        predictor.load_models(model_dir)
        logger.info(f"Loaded pre-trained models from {model_dir}")
    except Exception as e:
        logger.warning(f"Could not load pre-trained models: {e}")
        logger.info("Training new models on historical data")
        predictor.train(historical_data, use_ensemble=True, use_bayesian=True)
    
    # Make predictions
    logger.info(f"Making predictions with safety_car_prob={safety_car_prob}, rain_prob={rain_prob}")
    
    monte_carlo_data = None
    if use_monte_carlo:
        # Run Monte Carlo simulation for more robust predictions with uncertainty
        predictions = predictor.predict(
            quali_data=quali_data,
            driver_stats=driver_stats,
            team_stats=team_stats,
            safety_car_prob=safety_car_prob,
            rain_prob=rain_prob,
            use_monte_carlo=True,
            return_uncertainty=True
        )
        
        # Extract main predictions and monte carlo data
        if isinstance(predictions, dict):
            prediction_df = predictions['predictions']
            finishing_stats = predictions['finishing_statistics']
            monte_carlo_data = {
                'finishing_statistics': finishing_stats,
                'simulation_results': predictions['simulation_results']
            }
            if 'position_probabilities' in predictions:
                monte_carlo_data['probability_matrix'] = predictions['position_probabilities']
            logger.info("Generated race predictions with uncertainty using Monte Carlo simulation")
        else:
            prediction_df = predictions
            finishing_stats = None
            logger.info("Generated race predictions")
    else:
        # Standard prediction
        prediction_df = predictor.predict(
            quali_data=quali_data,
            driver_stats=driver_stats,
            team_stats=team_stats,
            safety_car_prob=safety_car_prob,
            rain_prob=rain_prob
        )
        finishing_stats = None
        logger.info("Generated race predictions")
    
    # Create output directory
    results_dir = ensure_directory('results')
    race_dir = ensure_directory(os.path.join(results_dir, f"round_{round_num}_{race_name.replace(' ', '_')}"))
    viz_dir = ensure_directory(os.path.join(race_dir, "visualizations"))
    
    # Import visualization modules
    try:
        # Basic visualizations
        from visualization.base import format_output
        from visualization.race import (
            plot_grid_vs_finish, 
            plot_predictions, 
            plot_top10_intervals, 
            plot_position_changes
        )
        from visualization.team import plot_team_performance, plot_team_pace_comparison
        
        logger.info("Loaded visualization modules")
        visualization_available = True
    except ImportError as e:
        logger.warning(f"Could not import visualization modules: {e}")
        logger.warning("Using basic visualizations only")
        visualization_available = False
    
    # Create and save visualizations
    visualization_paths = {}
    
    # 1. Basic visualizations always available
    try:
        # Grid vs Finish position plot
        fig_grid = plot_grid_vs_finish(
            prediction_df,
            title=f"Grid vs Finish Positions - {race_name}",
            save_path=os.path.join(viz_dir, "grid_vs_finish.png")
        )
        visualization_paths['grid_vs_finish'] = os.path.join(viz_dir, "grid_vs_finish.png")
        plt.close(fig_grid)
        
        # Top 10 finishers visualization
        fig_top10 = plot_predictions(
            prediction_df, 
            save_path=os.path.join(viz_dir, "top10_predictions.png"),
            show_plot=False
        )
        visualization_paths['top10'] = os.path.join(viz_dir, "top10_predictions.png")
        plt.close(fig_top10)
        
        # Team performance visualization
        fig_team = plot_team_performance(
            prediction_df, 
            save_path=os.path.join(viz_dir, "team_performance.png")
        )
        visualization_paths['team_performance'] = os.path.join(viz_dir, "team_performance.png")
        plt.close(fig_team)
        
        logger.info("Created basic visualizations")
    except Exception as e:
        logger.error(f"Error creating basic visualizations: {e}")
    
    # 2. Advanced visualizations if available
    if visualization_available:
        try:
            # Team pace comparison
            team_pace_path = plot_team_pace_comparison(
                prediction_df, 
                folder_path=viz_dir
            )
            visualization_paths['team_pace'] = team_pace_path
            logger.info("Created team pace comparison visualization")
            
            # Position changes simulation
            position_changes_path = plot_position_changes(
                prediction_df, 
                folder_path=viz_dir
            )
            visualization_paths['position_changes'] = position_changes_path
            logger.info("Created position changes visualization")
            
            # Top driver comparisons
            if len(prediction_df) >= 2:
                try:
                    from visualization.driver import create_driver_comparison
                    
                    # Compare top 2 drivers
                    top_driver1 = prediction_df.iloc[0]['Driver']
                    top_driver2 = prediction_df.iloc[1]['Driver']
                    
                    create_driver_comparison(
                        prediction_df,
                        top_driver1,
                        top_driver2,
                        historical_data=historical_data,
                        save_path=os.path.join(viz_dir, f"{top_driver1.lower()}_{top_driver2.lower()}_comparison.png")
                    )
                    visualization_paths['driver_comparison'] = os.path.join(viz_dir, f"{top_driver1.lower()}_{top_driver2.lower()}_comparison.png")
                    logger.info(f"Created driver comparison visualization for {top_driver1} vs {top_driver2}")
                except Exception as driver_e:
                    logger.warning(f"Could not create driver comparison: {driver_e}")
            
            # Add Monte Carlo specific visualizations if available
            if monte_carlo_data is not None and use_monte_carlo:
                try:
                    # Try to access the Monte Carlo module directly
                    from models.monte_carlo import MonteCarloRaceSimulator
                    simulator = MonteCarloRaceSimulator(predictor)
                    
                    # Create position distributions visualization
                    sim_results = monte_carlo_data['simulation_results']
                    
                    # Plot top drivers position distributions
                    fig_positions = simulator.plot_position_distributions(
                        sim_results, top_n_drivers=5
                    )
                    positions_path = os.path.join(viz_dir, "position_distributions.png")
                    fig_positions.savefig(positions_path)
                    visualization_paths['position_distributions'] = positions_path
                    plt.close(fig_positions)
                    
                    # Plot heatmap of finishing positions
                    fig_heatmap = simulator.plot_finishing_heatmap(
                        sim_results, top_n_drivers=10
                    )
                    heatmap_path = os.path.join(viz_dir, "finishing_heatmap.png")
                    fig_heatmap.savefig(heatmap_path)
                    visualization_paths['finishing_heatmap'] = heatmap_path
                    plt.close(fig_heatmap)
                    
                    logger.info("Created Monte Carlo visualizations")
                except Exception as mc_e:
                    logger.warning(f"Could not create Monte Carlo visualizations: {mc_e}")
                    
                    # Alternative: try position distribution visualization from race module
                    try:
                        if 'probability_matrix' in monte_carlo_data:
                            from visualization.race import plot_position_distribution
                            prob_matrix = monte_carlo_data['probability_matrix']
                            
                            # Plot overall position distributions
                            plot_position_distribution(
                                prob_matrix,
                                save_path=os.path.join(viz_dir, "position_distributions.png")
                            )
                            visualization_paths['position_distributions'] = os.path.join(viz_dir, "position_distributions.png")
                            
                            # Plot for top driver
                            top_driver = prediction_df.iloc[0]['Driver']
                            plot_position_distribution(
                                prob_matrix,
                                driver=top_driver,
                                save_path=os.path.join(viz_dir, f"{top_driver.lower()}_position_distribution.png")
                            )
                            visualization_paths[f'{top_driver}_position'] = os.path.join(viz_dir, f"{top_driver.lower()}_position_distribution.png")
                            
                            logger.info("Created alternative position distribution visualizations")
                    except Exception as alt_e:
                        logger.warning(f"Could not create alternative position visualizations: {alt_e}")
            
            # Try to create a dashboard visualization
            try:
                from visualization.dashboard import create_race_report, create_interactive_dashboard
                
                # Create dashboard
                dashboard_path = create_interactive_dashboard(
                    prediction_df,
                    track_name=track_name,
                    save_path=os.path.join(viz_dir, "dashboard.png"),
                    additional_data={'race_info': race_info}
                )
                visualization_paths['dashboard'] = dashboard_path
                logger.info("Created dashboard visualization")
                
                # Create HTML report with all visualizations
                additional_data = {
                    'race_info': {
                        'round': round_num,
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'track': track_name,
                        'name': race_name
                    }
                }
                
                if monte_carlo_data is not None:
                    additional_data['monte_carlo'] = monte_carlo_data
                
                report_path = create_race_report(
                    prediction_df,
                    track_name=track_name,
                    monte_carlo_data=monte_carlo_data,
                    save_path=os.path.join(race_dir, "race_report.html")
                )
                visualization_paths['report'] = report_path
                logger.info("Created comprehensive race report")
            except Exception as dash_e:
                logger.warning(f"Could not create dashboard or report: {dash_e}")
                
                # Try simpler dashboard if available
                try:
                    from visualization.dashboard import create_simple_dashboard
                    
                    simple_dash_path = create_simple_dashboard(
                        prediction_df,
                        track_name=track_name,
                        save_path=os.path.join(viz_dir, "simple_dashboard.png")
                    )
                    visualization_paths['simple_dashboard'] = simple_dash_path
                    logger.info("Created simple dashboard visualization")
                except Exception as simple_e:
                    logger.warning(f"Could not create simple dashboard: {simple_e}")
                    
        except Exception as adv_e:
            logger.error(f"Error creating advanced visualizations: {adv_e}")
    
    # Analyze tire strategy if available
    try:
        from visualization.tire import compare_compounds, plot_crossover_analysis
        
        # Try to create tire strategy visualizations
        try:
            # Tire compound comparison
            compound_comparison = compare_compounds(
                track_name,
                save_path=os.path.join(viz_dir, "compound_comparison.png")
            )
            visualization_paths['compound_comparison'] = os.path.join(viz_dir, "compound_comparison.png")
            logger.info("Created tire compound comparison")
            
            # Crossover analysis
            crossovers = plot_crossover_analysis(
                track_name,
                save_path=os.path.join(viz_dir, "crossover_analysis.png")
            )
            visualization_paths['crossover_analysis'] = os.path.join(viz_dir, "crossover_analysis.png")
            logger.info("Created tire crossover analysis")
            
            # Add tire data to prediction_df for reference
            if compound_comparison and 'optimal_stints' in compound_comparison:
                tire_data = {
                    'optimal_stints': compound_comparison['optimal_stints'],
                    'crossovers': crossovers
                }
            else:
                tire_data = None
        except Exception as tire_e:
            logger.warning(f"Could not create tire strategy visualizations: {tire_e}")
            tire_data = None
    except ImportError:
        logger.warning("Tire visualization modules not available")
        tire_data = None
    
    # Save predictions to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(race_dir, f"{timestamp}_predictions.csv")
    prediction_df.to_csv(csv_filename, index=False)
    logger.info(f"Saved predictions to {csv_filename}")
    
    # If Monte Carlo was used, save detailed stats
    if use_monte_carlo and finishing_stats is not None:
        stats_filename = os.path.join(race_dir, f"{timestamp}_finishing_statistics.csv")
        finishing_stats.to_csv(stats_filename, index=False)
        logger.info(f"Saved finishing statistics to {stats_filename}")

    results_dir = ensure_directory('results')
    race_dir = ensure_directory(os.path.join(results_dir, f"round_{round_num}_{race_name.replace(' ', '_')}"))
    
    # Generate visualizations
    logger.info("Creating standard visualizations")
    from visualization.race import plot_grid_vs_finish, plot_top10_intervals, plot_position_changes
    from visualization.team import plot_team_performance, plot_team_pace_comparison
    
    # Grid vs Finish
    fig_grid = plot_grid_vs_finish(
        prediction_df, 
        title=f"Grid vs Race Position - {race_name}",
        save_path=os.path.join(race_dir, "grid_vs_finish.png")
    )
    plt.close(fig_grid)
    
    # Top 10 Intervals
    top10_path = plot_top10_intervals(prediction_df, folder_path=race_dir)
    
    # Position Changes
    position_path = plot_position_changes(prediction_df, folder_path=race_dir)
    
    # Team Performance
    team_path = plot_team_performance(
        prediction_df, 
        save_path=os.path.join(race_dir, "team_performance.png")
    )
    
    # Team Pace Comparison
    pace_path = plot_team_pace_comparison(prediction_df, folder_path=race_dir)
    
    # Format text output using the format_output function if available
    try:
        from visualization.base import format_output
        formatted_output = format_output(prediction_df)
    except (ImportError, AttributeError):
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
    
    # Print top 10 to console using the formatted output
    print(formatted_output)
    
    # Print probabilities if available
    if use_monte_carlo and finishing_stats is not None:
        print("\nWin Probabilities:")
        win_probs = finishing_stats[['Driver', 'WinProbability']].sort_values('WinProbability', ascending=False).head(5)
        for _, row in win_probs.iterrows():
            print(f"{row['Driver']}: {row['WinProbability'] * 100:.1f}%")
    
    # Print visualization paths
    if visualization_paths:
        print("\nVisualizations created:")
        for name, path in visualization_paths.items():
            print(f"- {name}: {path}")
    
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