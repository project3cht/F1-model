"""
Integration of the updated F1Schedule with the prediction service.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('f1_prediction_integration')

# Import our modules
from utils.constants import (
    f1_schedule,
    get_race_info,
    get_upcoming_races,
    find_race_by_name,
    TRACK_CHARACTERISTICS,
    TEAM_COLORS,
    DRIVERS
)
from features.feature_store_factory import FeatureStoreFactory
from config.config import load_config
from data.fetching import load_sample_data
from application.prediction_service import PredictionService
from domain.circuit import Circuit
from visualization.base import plot_grid_vs_finish, save_figure
from utils.helpers import ensure_directory

def make_prediction_for_round(round_num, year=2025, use_monte_carlo=True, config_path="config/config.yaml"):
    """
    Make predictions for a specific race using the enhanced F1Schedule data.
    
    Args:
        round_num: Round number
        year: Season year
        use_monte_carlo: Whether to use Monte Carlo simulation
        config_path: Path to configuration file
        
    Returns:
        DataFrame: Race predictions
    """
    logger.info(f"Making prediction for Round {round_num}, {year}")
    
    # Get race information from schedule
    race_info = get_race_info(round_num, year)
    
    if not race_info:
        logger.error(f"Could not find race info for Round {round_num}, {year}")
        return None
    
    # Display race information
    logger.info(f"Race: {race_info['name']}")
    logger.info(f"Circuit: {race_info['track']}")
    logger.info(f"Format: {race_info['format']}")
    logger.info(f"Has Sprint: {race_info['has_sprint']}")
    
    # Get safety car and rain probabilities from track characteristics
    safety_car_prob = race_info['safety_car_prob']
    rain_prob = race_info['rain_prob']
    
    logger.info(f"Safety Car Probability: {safety_car_prob:.2f}")
    logger.info(f"Rain Probability: {rain_prob:.2f}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Create feature store
    feature_store = FeatureStoreFactory.create_feature_store()
    logger.info("Initialized feature store")
    
    # Create prediction service
    prediction_service = PredictionService(config_path=config_path)
    logger.info("Initialized prediction service")
    
    # Create Circuit entity
    circuit = Circuit.create(
        circuit_id=f"circuit_{round_num}",
        name=race_info['track'],
        country=race_info['country']
    )
    
    # For this example, load sample qualifying data or simulate it
    quali_data = load_sample_data(data_type='qualifying')
    logger.info(f"Loaded qualifying data with {len(quali_data)} drivers")
    
    # Add track name to qualifying data for track-specific features
    quali_data['Track'] = race_info['track']
    
    # Load sample historical data for context
    historical_data = load_sample_data(data_type='race')
    logger.info(f"Loaded historical data with {len(historical_data)} entries")
    
    # Make predictions using the prediction service
    prediction = prediction_service.predict_race(
        race_id=f"race_{year}_{round_num}",
        circuit=circuit,
        quali_data=quali_data,
        historical_data=historical_data,
        safety_car_prob=safety_car_prob,
        rain_prob=rain_prob,
        use_monte_carlo=use_monte_carlo
    )
    
    # Extract results from prediction
    if use_monte_carlo:
        logger.info("Generated race predictions with Monte Carlo simulation")
        results = []
        
        # Convert prediction results to DataFrame
        for result in prediction.results:
            results.append({
                'Driver': result.driver.name,
                'Team': result.driver.team.name,
                'Position': result.predicted_position,
                'Grid': quali_data[quali_data['Driver'] == result.driver.name]['GridPosition'].values[0],
                'Interval': "WINNER" if result.predicted_position == 1 else f"+{result.interval_from_leader:.3f}s",
                'Interval (s)': result.interval_from_leader if result.interval_from_leader else 0,
                'Points': result.points if hasattr(result, 'points') else 0,
                'Confidence': result.confidence_score
            })
    else:
        logger.info("Generated race predictions")
        results = []
        
        # Convert prediction results to DataFrame
        for result in prediction.results:
            results.append({
                'Driver': result.driver.name,
                'Team': result.driver.team.name,
                'Position': result.predicted_position,
                'Grid': quali_data[quali_data['Driver'] == result.driver.name]['GridPosition'].values[0],
                'Interval': "WINNER" if result.predicted_position == 1 else f"+{result.interval_from_leader:.3f}s",
                'Interval (s)': result.interval_from_leader if result.interval_from_leader else 0,
                'Points': result.points if hasattr(result, 'points') else 0
            })
    
    # Create DataFrame
    predictions_df = pd.DataFrame(results)
    
    # Sort by position
    predictions_df = predictions_df.sort_values('Position').reset_index(drop=True)
    
    # Create output directory
    results_dir = ensure_directory('results')
    race_dir = ensure_directory(os.path.join(results_dir, f"round_{round_num}_{race_info['name'].replace(' ', '_')}"))
    
    # Create visualizations
    fig = plot_grid_vs_finish(
        predictions_df, 
        title=f"Grid vs. Finish - {race_info['name']}"
    )
    
    # Save visualizations
    save_path = os.path.join(race_dir, "grid_vs_finish.png")
    plt.savefig(save_path)
    plt.close(fig)
    
    # Save predictions to CSV
    csv_path = os.path.join(race_dir, "predictions.csv")
    predictions_df.to_csv(csv_path, index=False)
    
    # Return predictions
    return predictions_df

def predict_upcoming_races(count=3, use_monte_carlo=True, config_path="config/config.yaml"):
    """
    Make predictions for upcoming races on the calendar.
    
    Args:
        count: Number of upcoming races to predict
        use_monte_carlo: Whether to use Monte Carlo simulation
        config_path: Path to configuration file
        
    Returns:
        dict: Dictionary of race predictions by round number
    """
    logger.info(f"Predicting upcoming {count} races")
    
    # Get upcoming races
    upcoming = get_upcoming_races(count=count)
    
    # Make predictions for each upcoming race
    predictions = {}
    
    for _, row in upcoming.iterrows():
        # Get round number
        if 'RoundNumber' in row:
            round_num = row['RoundNumber']
        else:
            round_num = row['round']
        
        # Make prediction
        try:
            prediction = make_prediction_for_round(
                round_num=round_num,
                use_monte_carlo=use_monte_carlo,
                config_path=config_path
            )
            
            predictions[round_num] = prediction
            logger.info(f"Successfully predicted Round {round_num}")
            
        except Exception as e:
            logger.error(f"Error predicting Round {round_num}: {e}")
    
    return predictions

def compare_sprint_vs_regular(year=2025, config_path="config/config.yaml"):
    """
    Compare predictions for sprint races vs regular race weekends.
    
    Args:
        year: Season year
        config_path: Path to configuration file
        
    Returns:
        tuple: DataFrames for sprint and regular race predictions
    """
    logger.info("Comparing sprint vs regular race weekend predictions")
    
    # Get full schedule
    schedule = f1_schedule.get_event_schedule(year)
    
    # Find races with different formats
    sprint_rounds = []
    regular_rounds = []
    
    # Check each round
    for i, row in schedule.iterrows():
        # Get round number
        if 'RoundNumber' in row:
            round_num = row['RoundNumber']
        else:
            round_num = row['round']
        
        # Check format
        format_type = f1_schedule.event_formats.get(round_num, 'conventional')
        
        if 'sprint' in format_type:
            sprint_rounds.append(round_num)
        else:
            regular_rounds.append(round_num)
    
    # Select one sprint and one regular race for comparison
    sprint_round = sprint_rounds[0] if sprint_rounds else None
    regular_round = regular_rounds[0] if regular_rounds else None
    
    # Make predictions
    sprint_prediction = None
    regular_prediction = None
    
    if sprint_round:
        try:
            sprint_prediction = make_prediction_for_round(
                round_num=sprint_round,
                use_monte_carlo=False,
                config_path=config_path
            )
            logger.info(f"Made prediction for sprint round {sprint_round}")
        except Exception as e:
            logger.error(f"Error predicting sprint round {sprint_round}: {e}")
    
    if regular_round:
        try:
            regular_prediction = make_prediction_for_round(
                round_num=regular_round,
                use_monte_carlo=False,
                config_path=config_path
            )
            logger.info(f"Made prediction for regular round {regular_round}")
        except Exception as e:
            logger.error(f"Error predicting regular round {regular_round}: {e}")
    
    # Compare predictions
    if sprint_prediction is not None and regular_prediction is not None:
        # Create output directory
        results_dir = ensure_directory('results')
        comparison_dir = ensure_directory(os.path.join(results_dir, 'format_comparison'))
        
        # Create comparison visualization
        plt.figure(figsize=(12, 8))
        
        # Merge predictions by driver
        merged = pd.merge(
            sprint_prediction[['Driver', 'Position']],
            regular_prediction[['Driver', 'Position']],
            on='Driver',
            suffixes=('_Sprint', '_Regular')
        )
        
        # Plot position differences
        plt.scatter(merged['Position_Sprint'], merged['Position_Regular'], alpha=0.7)
        
        # Add diagonal line
        plt.plot([1, 20], [1, 20], 'k--', alpha=0.5)

        # Add driver labels
        for _, row in merged.iterrows():
            plt.annotate(
                row['Driver'],
                (row['Position_Sprint'], row['Position_Regular']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
        
        # Format plot
        plt.title('Sprint vs Regular Race Weekend Position Comparison', fontsize=14)
        plt.xlabel('Position in Sprint Weekend', fontsize=12)
        plt.ylabel('Position in Regular Weekend', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().invert_xaxis()  # Invert axes so that P1 is at top-left
        plt.gca().invert_yaxis()
        
        # Save figure
        comparison_path = os.path.join(comparison_dir, 'format_comparison.png')
        plt.savefig(comparison_path)
        plt.close()
        
        logger.info(f"Saved format comparison to {comparison_path}")
        
        # Save comparison data
        csv_path = os.path.join(comparison_dir, 'format_comparison.csv')
        merged.to_csv(csv_path, index=False)
        
        logger.info(f"Saved comparison data to {csv_path}")
        
    return sprint_prediction, regular_prediction

def analyze_circuit_effect(year=2025, config_path="config/config.yaml"):
    """
    Analyze how different circuit characteristics affect predictions.
    
    Args:
        year: Season year
        config_path: Path to configuration file
        
    Returns:
        DataFrame: Summary of circuit effects
    """
    logger.info("Analyzing circuit effects on predictions")
    
    # Get schedule
    schedule = f1_schedule.get_event_schedule(year)
    
    # Select a few different circuit types
    circuit_types = {
        'high_speed': None,
        'technical': None,
        'street': None
    }
    
    # Find races with different circuit types
    for i, row in schedule.iterrows():
        # Get round number and track
        if 'RoundNumber' in row:
            round_num = row['RoundNumber']
            track = row['Location']
        else:
            round_num = row['round']
            track = row['circuit']
        
        # Match track to track characteristics
        track_key = f1_schedule._find_matching_track(track)
        track_chars = TRACK_CHARACTERISTICS.get(track_key, {})
        
        # Classify circuit type (simplified classification)
        if track_chars.get('layout_stress', 0) > 0.8:
            if circuit_types['high_speed'] is None:
                circuit_types['high_speed'] = round_num
        elif 'Monaco' in track or 'Singapore' in track or 'Baku' in track:
            if circuit_types['street'] is None:
                circuit_types['street'] = round_num
        elif track_chars.get('layout_stress', 0) > 0.5:
            if circuit_types['technical'] is None:
                circuit_types['technical'] = round_num
    
    # Make predictions for each circuit type
    circuit_predictions = {}
    
    for circuit_type, round_num in circuit_types.items():
        if round_num is not None:
            try:
                prediction = make_prediction_for_round(
                    round_num=round_num,
                    use_monte_carlo=False,
                    config_path=config_path
                )
                circuit_predictions[circuit_type] = prediction
                logger.info(f"Made prediction for {circuit_type} circuit (Round {round_num})")
            except Exception as e:
                logger.error(f"Error predicting {circuit_type} circuit (Round {round_num}): {e}")
    
    # Analyze prediction differences
    if len(circuit_predictions) > 1:
        # Create output directory
        results_dir = ensure_directory('results')
        analysis_dir = ensure_directory(os.path.join(results_dir, 'circuit_analysis'))
        
        # Create summary DataFrame
        summary_data = []
        
        # Analyze top driver performances across circuits
        top_drivers = set()
        for circuit_type, prediction in circuit_predictions.items():
            # Get top 5 drivers
            top_5 = prediction.head(5)['Driver'].values
            top_drivers.update(top_5)
        
        # Create comparison data for each top driver
        for driver in top_drivers:
            driver_data = {'Driver': driver}
            
            # Get position in each circuit type
            for circuit_type, prediction in circuit_predictions.items():
                if driver in prediction['Driver'].values:
                    pos = prediction[prediction['Driver'] == driver]['Position'].values[0]
                    driver_data[f'{circuit_type}_pos'] = pos
                else:
                    driver_data[f'{circuit_type}_pos'] = None
            
            # Add to summary
            summary_data.append(driver_data)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Set up bar positions
        n_drivers = len(summary_df)
        n_circuits = len(circuit_types)
        bar_width = 0.8 / n_circuits
        
        # Plot positions for each circuit type
        for i, circuit_type in enumerate(circuit_types.keys()):
            if f'{circuit_type}_pos' in summary_df.columns:
                positions = summary_df[f'{circuit_type}_pos'].values
                x_pos = np.arange(n_drivers) + i * bar_width - (n_circuits - 1) * bar_width / 2
                
                plt.bar(
                    x_pos,
                    positions,
                    width=bar_width,
                    label=circuit_type.replace('_', ' ').title()
                )
        
        # Format plot
        plt.title('Driver Performance Across Different Circuit Types', fontsize=14)
        plt.xlabel('Driver', fontsize=12)
        plt.ylabel('Position', fontsize=12)
        plt.xticks(np.arange(n_drivers), summary_df['Driver'], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.gca().invert_yaxis()  # Invert y-axis so position 1 is at the top
        
        # Save figure
        comparison_path = os.path.join(analysis_dir, 'circuit_comparison.png')
        plt.tight_layout()
        plt.savefig(comparison_path)
        plt.close()
        
        logger.info(f"Saved circuit comparison to {comparison_path}")
        
        # Save comparison data
        csv_path = os.path.join(analysis_dir, 'circuit_comparison.csv')
        summary_df.to_csv(csv_path, index=False)
        
        logger.info(f"Saved circuit comparison data to {csv_path}")
        
        return summary_df
    
    return None

def main():
    """Main function demonstrating integration of F1Schedule with prediction service."""
    # Set the config path
    config_path = "config/config.yaml"
    
    # Make a prediction for a specific round
    print("\n=== MAKING PREDICTION FOR ROUND 7 ===")
    predictions = make_prediction_for_round(
        round_num=7,
        use_monte_carlo=False,
        config_path=config_path
    )
    
    # Display top 10 predictions
    if predictions is not None:
        print("\nTop 10 Predicted Finishers:")
        for i, row in predictions.head(10).iterrows():
            print(f"{int(row['Position']):2d}. {row['Driver']:<20} ({row['Team']:<15}) | Grid: {int(row['Grid']):2d} | {row['Interval']}")
    
    # Compare sprint vs regular race predictions
    print("\n=== COMPARING SPRINT VS REGULAR RACE FORMATS ===")
    sprint_pred, regular_pred = compare_sprint_vs_regular(
        year=2025,
        config_path=config_path
    )
    
    # Analyze circuit effects
    print("\n=== ANALYZING CIRCUIT EFFECTS ===")
    circuit_analysis = analyze_circuit_effect(
        year=2025,
        config_path=config_path
    )

if __name__ == "__main__":
    main()