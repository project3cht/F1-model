# models/monte_carlo.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Callable
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
from concurrent.futures import ProcessPoolExecutor
import os

class MonteCarloRaceSimulator:
    """
    Monte Carlo race simulator for F1 predictions with uncertainty estimates.
    """
    
    def __init__(self, 
                base_predictor: Any,
                n_simulations: int = 1000,
                safety_car_variations: bool = True,
                weather_variations: bool = True,
                use_multiprocessing: bool = True) -> None:
        """
        Initialize Monte Carlo race simulator.
        
        Args:
            base_predictor: Base prediction model
            n_simulations: Number of simulations to run
            safety_car_variations: Whether to vary safety car probability
            weather_variations: Whether to vary weather conditions
            use_multiprocessing: Whether to use multiprocessing
        """
        self.base_predictor = base_predictor
        self.n_simulations = n_simulations
        self.safety_car_variations = safety_car_variations
        self.weather_variations = weather_variations
        self.use_multiprocessing = use_multiprocessing
        
    def _run_single_simulation(self, 
                              quali_data: pd.DataFrame,
                              seed: int,
                              base_safety_car_prob: float = 0.6,
                              base_rain_prob: float = 0.0) -> pd.DataFrame:
        """
        Run a single race simulation.
        
        Args:
            quali_data: Qualifying data
            seed: Random seed for reproducibility
            base_safety_car_prob: Base safety car probability
            base_rain_prob: Base rain probability
            
        Returns:
            Race result for this simulation
        """
        # Set seed for reproducibility
        np.random.seed(seed)
        
        # Vary safety car probability if enabled
        safety_car_prob = base_safety_car_prob
        if self.safety_car_variations:
            # Add random variation
            safety_car_prob = np.clip(
                np.random.normal(base_safety_car_prob, 0.15),
                0.0, 1.0
            )
        
        # Vary rain probability if enabled
        rain_prob = base_rain_prob
        if self.weather_variations:
            # Add random variation, more likely to increase than decrease
            if np.random.random() < 0.7 and base_rain_prob < 0.8:
                # Increase rain probability
                rain_increase = np.random.beta(2, 5)  # Beta distribution skewed to smaller increases
                rain_prob = min(1.0, base_rain_prob + rain_increase)
            else:
                # Decrease rain probability
                rain_prob = max(0.0, base_rain_prob * np.random.uniform(0.5, 1.0))
        
        # Make prediction with varied parameters
        result = self.base_predictor.predict(
            quali_data=quali_data,
            safety_car_prob=safety_car_prob,
            rain_prob=rain_prob
        )
        
        # Add simulation metadata
        result['SimulationId'] = seed
        result['SafetyCarProb'] = safety_car_prob
        result['RainProb'] = rain_prob
        
        return result
    
    def simulate_race(self, 
                     quali_data: pd.DataFrame,
                     base_safety_car_prob: float = 0.6,
                     base_rain_prob: float = 0.0) -> pd.DataFrame:
        """
        Run multiple race simulations with Monte Carlo method.
        
        Args:
            quali_data: Qualifying data
            base_safety_car_prob: Base safety car probability
            base_rain_prob: Base rain probability
            
        Returns:
            Combined race results from all simulations
        """
        if self.use_multiprocessing:
            # Use multiprocessing for faster simulation
            seeds = list(range(self.n_simulations))
            
            with ProcessPoolExecutor() as executor:
                results = list(tqdm(
                    executor.map(
                        lambda seed: self._run_single_simulation(
                            quali_data, seed, base_safety_car_prob, base_rain_prob
                        ),
                        seeds
                    ),
                    total=self.n_simulations,
                    desc="Running simulations"
                ))
        else:
            # Single process simulation
            results = []
            for seed in tqdm(range(self.n_simulations), desc="Running simulations"):
                result = self._run_single_simulation(
                    quali_data, seed, base_safety_car_prob, base_rain_prob
                )
                results.append(result)
        
        # Combine all simulation results
        all_results = pd.concat(results, ignore_index=True)
        
        return all_results
    
    def calculate_position_probabilities(self, simulation_results: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate probabilities of each driver finishing in each position.
        
        Args:
            simulation_results: Combined results from all simulations
            
        Returns:
            DataFrame with position probabilities
        """
        # Count occurrences of each driver in each position
        position_counts = pd.crosstab(
            simulation_results['Driver'], 
            simulation_results['Position']
        )
        
        # Convert to probabilities
        position_probs = position_counts / self.n_simulations
        
        return position_probs
    
    def calculate_finishing_statistics(self, simulation_results: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate finishing statistics for each driver.
        
        Args:
            simulation_results: Combined results from all simulations
            
        Returns:
            DataFrame with finishing statistics
        """
        # Group by driver
        driver_stats = simulation_results.groupby('Driver').agg({
            'Position': ['mean', 'std', 'min', 'max'],
            'Points': ['mean', 'std', 'min', 'max']
        })
        
        # Flatten multi-index columns
        driver_stats.columns = [f"{col[0]}_{col[1]}" for col in driver_stats.columns]
        
        # Calculate probability of podium (positions 1-3)
        podium_probs = simulation_results[simulation_results['Position'] <= 3].groupby('Driver').size() / self.n_simulations
        driver_stats['PodiumProbability'] = podium_probs
        
        # Calculate probability of points (positions 1-10)
        points_probs = simulation_results[simulation_results['Position'] <= 10].groupby('Driver').size() / self.n_simulations
        driver_stats['PointsProbability'] = points_probs
        
        # Calculate win probability (position 1)
        win_probs = simulation_results[simulation_results['Position'] == 1].groupby('Driver').size() / self.n_simulations
        driver_stats['WinProbability'] = win_probs
        
        # Reset index to make Driver a column
        driver_stats = driver_stats.reset_index()
        
        # Fill NaN values with 0 (for drivers who never achieved certain results)
        driver_stats = driver_stats.fillna(0)
        
        return driver_stats.sort_values('Position_mean')
    
    def plot_position_distributions(self, simulation_results: pd.DataFrame, 
                                   top_n_drivers: int = 5) -> plt.Figure:
        """
        Plot position distribution for top drivers.
        
        Args:
            simulation_results: Combined results from all simulations
            top_n_drivers: Number of top drivers to include
            
        Returns:
            Matplotlib figure
        """
        # Get average positions to determine top drivers
        avg_positions = simulation_results.groupby('Driver')['Position'].mean().sort_values()
        top_drivers = avg_positions.index[:top_n_drivers].tolist()
        
        # Filter data for top drivers
        top_driver_results = simulation_results[simulation_results['Driver'].isin(top_drivers)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create violin plot
        sns.violinplot(
            data=top_driver_results,
            x='Driver',
            y='Position',
            ax=ax,
            inner='quartile',
            order=top_drivers
        )
        
        # Invert y-axis so 1st position is at the top
        ax.invert_yaxis()
        
        # Set title and labels
        ax.set_title('Position Distribution for Top Drivers', fontsize=14)
        ax.set_xlabel('Driver', fontsize=12)
        ax.set_ylabel('Position', fontsize=12)
        
        # Add a grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add text with win probabilities
        for i, driver in enumerate(top_drivers):
            win_prob = (simulation_results[simulation_results['Driver'] == driver]['Position'] == 1).mean() * 100
            podium_prob = (simulation_results[simulation_results['Driver'] == driver]['Position'] <= 3).mean() * 100
            
            ax.text(
                i, ax.get_ylim()[0] + 0.5,
                f"Win: {win_prob:.1f}%\nPodium: {podium_prob:.1f}%",
                ha='center',
                va='top',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3')
            )
        
        plt.tight_layout()
        return fig
    
    def plot_finishing_heatmap(self, simulation_results: pd.DataFrame,
                              top_n_drivers: int = 10) -> plt.Figure:
        """
        Plot heatmap of position probabilities.
        
        Args:
            simulation_results: Combined results from all simulations
            top_n_drivers: Number of top drivers to include
            
        Returns:
            Matplotlib figure
        """
        # Get average positions to determine top drivers
        avg_positions = simulation_results.groupby('Driver')['Position'].mean().sort_values()
        top_drivers = avg_positions.index[:top_n_drivers].tolist()
        
        # Calculate position probabilities
        position_probs = self.calculate_position_probabilities(simulation_results)
        
        # Filter for top drivers and positions 1-10
        heatmap_data = position_probs.loc[top_drivers, range(1, 11)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.2f',
            cmap='YlGnBu',
            ax=ax,
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Probability'}
        )
        
        # Set title and labels
        ax.set_title('Position Probability Heatmap', fontsize=14)
        ax.set_xlabel('Position', fontsize=12)
        ax.set_ylabel('Driver', fontsize=12)
        
        plt.tight_layout()
        return fig