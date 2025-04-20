# application/analysis_service.py
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

from domain.driver import Driver, DriverPerformance
from domain.team import Team, TeamPerformance
from domain.race import Race, RaceResult
from domain.circuit import Circuit
from data.processing import calculate_driver_stats, calculate_team_stats
from visualization.plots import plot_grid_vs_finish, plot_race_results
from utils.helpers import ensure_directory, save_figure

class AnalysisService:
    """Application service for F1 race analysis."""
    
    def __init__(self):
        """Initialize analysis service."""
        self.logger = logging.getLogger('f1_prediction.analysis_service')
        self.results_dir = ensure_directory('results')
        
    def analyze_race_performance(self, race: Race) -> Dict[str, Any]:
        """Analyze performance from a completed race."""
        self.logger.info(f"Analyzing race {race.id.value}")
        
        analysis = {
            'winner': None,
            'driver_performances': {},
            'team_performances': {},
            'position_changes': {}
        }
        
        # Get race winner
        winner = race.get_winner()
        if winner:
            analysis['winner'] = winner.driver.name
        
        # Analyze position changes
        for result in race.results:
            if result.grid_position and not result.dnf:
                position_delta = result.grid_position - result.finishing_position
                analysis['position_changes'][result.driver.name] = position_delta
                
        # Group by team
        team_results = {}
        for result in race.results:
            team_name = result.driver.team.name
            if team_name not in team_results:
                team_results[team_name] = []
            team_results[team_name].append(result)
            
        # Analyze team performance
        for team_name, results in team_results.items():
            avg_position = sum(r.finishing_position for r in results) / len(results)
            total_points = sum(r.points_scored for r in results)
            analysis['team_performances'][team_name] = {
                'average_position': avg_position,
                'total_points': total_points
            }
        
        return analysis
    
    def analyze_driver_history(self, 
                             driver: Driver, 
                             races: List[Race]) -> DriverPerformance:
        """Analyze a driver's historical performance."""
        self.logger.info(f"Analyzing performance for driver {driver.name}")
        
        positions = []
        grid_positions = []
        points = []
        dnfs = 0
        wins = 0
        podiums = 0
        
        for race in races:
            for result in race.results:
                if result.driver.id.value == driver.id.value:
                    positions.append(result.finishing_position)
                    if result.grid_position:
                        grid_positions.append(result.grid_position)
                    points.append(result.points_scored)
                    
                    if result.dnf:
                        dnfs += 1
                    if result.finishing_position == 1:
                        wins += 1
                    if result.finishing_position <= 3:
                        podiums += 1
                    
                    break
        
        total_races = len(positions)
        if total_races == 0:
            return DriverPerformance()
        
        avg_finish = sum(positions) / total_races
        avg_grid = sum(grid_positions) / len(grid_positions) if grid_positions else 0
        positions_gained = sum((g - p) for g, p in zip(grid_positions, positions[:len(grid_positions)])) / len(grid_positions) if grid_positions else 0
        finishing_rate = (total_races - dnfs) / total_races
        win_rate = wins / total_races
        podium_rate = podiums / total_races
        
        return DriverPerformance(
            avg_finish_position=avg_finish,
            avg_grid_position=avg_grid,
            positions_gained=positions_gained,
            finishing_rate=finishing_rate,
            win_rate=win_rate,
            podium_rate=podium_rate
        )
    
    def analyze_circuit_trends(self, 
                             circuit: Circuit, 
                             races: List[Race]) -> Dict[str, Any]:
        """Analyze trends at a specific circuit."""
        self.logger.info(f"Analyzing trends for circuit {circuit.name}")
        
        circuit_races = [race for race in races if race.circuit.id.value == circuit.id.value]
        
        analysis = {
            'total_races': len(circuit_races),
            'winning_teams': {},
            'winning_drivers': {},
            'safety_car_frequency': 0,
            'average_position_changes': 0
        }
        
        total_position_changes = 0
        total_safety_cars = 0
        
        for race in circuit_races:
            winner = race.get_winner()
            if winner:
                # Track winning teams
                team_name = winner.driver.team.name
                if team_name not in analysis['winning_teams']:
                    analysis['winning_teams'][team_name] = 0
                analysis['winning_teams'][team_name] += 1
                
                # Track winning drivers
                driver_name = winner.driver.name
                if driver_name not in analysis['winning_drivers']:
                    analysis['winning_drivers'][driver_name] = 0
                analysis['winning_drivers'][driver_name] += 1
            
            # Count safety cars
            if race.safety_car_deployed:
                total_safety_cars += 1
            
            # Calculate position changes
            for result in race.results:
                if result.grid_position and not result.dnf:
                    total_position_changes += abs(result.grid_position - result.finishing_position)
        
        if circuit_races:
            analysis['safety_car_frequency'] = total_safety_cars / len(circuit_races)
            analysis['average_position_changes'] = total_position_changes / (len(circuit_races) * 20)  # Assume 20 drivers
        
        return analysis
    
    def generate_visualizations(self, 
                              race: Race, 
                              predictions: Optional[pd.DataFrame] = None) -> List[str]:
        """Generate visualizations for race analysis."""
        self.logger.info(f"Generating visualizations for race {race.id.value}")
        saved_files = []
        
        # Convert race results to DataFrame for visualization
        results_data = []
        for result in race.results:
            results_data.append({
                'Driver': result.driver.name,
                'Team': result.driver.team.name,
                'Position': result.finishing_position,
                'GridPosition': result.grid_position or 0,
                'Interval': f"+{result.interval_from_winner:.3f}s" if result.interval_from_winner else "WINNER",
                'IntervalSeconds': result.interval_from_winner or 0.0,
                'Points': result.points_scored
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Race results visualization
        fig_results = plot_race_results(results_df, title=f"Race Results - {race.circuit.name}")
        results_path = save_figure(fig_results, f"{race.circuit.name}_results.png", self.results_dir)
        saved_files.append(results_path)
        
        # If predictions provided, create comparison visualization
        if predictions is not None:
            fig_comparison = plot_grid_vs_finish(predictions, title=f"Predicted vs Actual - {race.circuit.name}")
            comparison_path = save_figure(fig_comparison, f"{race.circuit.name}_comparison.png", self.results_dir)
            saved_files.append(comparison_path)
        
        return saved_files