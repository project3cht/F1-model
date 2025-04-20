# models/bayesian_models.py
import pandas as pd
import numpy as np
import pymc3 as pm
import arviz as az
from typing import Dict, List, Optional, Union, Tuple
import matplotlib.pyplot as plt
import theano.tensor as tt
import pickle
import os

class BayesianRacePredictionModel:
    """
    Bayesian model for F1 race predictions with rigorous uncertainty quantification.
    """
    
    def __init__(self, 
                model_type: str = 'hierarchical',
                samples: int = 2000,
                tune: int = 1000,
                chains: int = 2) -> None:
        """
        Initialize Bayesian prediction model.
        
        Args:
            model_type: Type of Bayesian model ('hierarchical' or 'gp')
            samples: Number of posterior samples
            tune: Number of tuning samples
            chains: Number of MCMC chains
        """
        self.model_type = model_type
        self.samples = samples
        self.tune = tune
        self.chains = chains
        self.model = None
        self.trace = None
        self.driver_indices = None
        self.team_indices = None
        self.track_indices = None
        
    def _prepare_indices(self, data: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """
        Prepare indices for categorical variables.
        
        Args:
            data: Training data
            
        Returns:
            Dictionary of indices for categorical variables
        """
        # Create indices for drivers
        unique_drivers = data['Driver'].unique()
        driver_indices = {driver: i for i, driver in enumerate(unique_drivers)}
        
        # Create indices for teams
        unique_teams = data['Team'].unique()
        team_indices = {team: i for i, team in enumerate(unique_teams)}
        
        # Create indices for tracks if available
        track_indices = {}
        if 'Track' in data.columns:
            unique_tracks = data['Track'].unique()
            track_indices = {track: i for i, track in enumerate(unique_tracks)}
        
        return {
            'driver_indices': driver_indices,
            'team_indices': team_indices,
            'track_indices': track_indices
        }
    
    def _build_hierarchical_model(self, data: pd.DataFrame) -> pm.Model:
        """
        Build hierarchical Bayesian model for race predictions.
        
        Args:
            data: Training data
            
        Returns:
            PyMC3 model
        """
        # Prepare indices
        indices = self._prepare_indices(data)
        self.driver_indices = indices['driver_indices']
        self.team_indices = indices['team_indices']
        self.track_indices = indices['track_indices']
        
        # Convert categorical variables to indices
        driver_idx = np.array([self.driver_indices[d] for d in data['Driver']])
        team_idx = np.array([self.team_indices[t] for t in data['Team']])
        
        # Convert track if available
        track_idx = None
        if 'Track' in data.columns:
            track_idx = np.array([self.track_indices[t] for t in data['Track']])
        
        # Extract numerical features
        grid_position = data['GridPosition'].values
        
        # Extract target
        position = data['Position'].values
        
        # Build model
        with pm.Model() as model:
            # Global intercept
            intercept = pm.Normal('intercept', mu=0, sigma=10)
            
            # Grid position effect (stronger prior towards positive correlation)
            grid_effect = pm.Normal('grid_effect', mu=0.8, sigma=0.3)
            
            # Driver skill (hierarchical)
            driver_skill_mu = pm.Normal('driver_skill_mu', mu=0, sigma=1)
            driver_skill_sigma = pm.HalfNormal('driver_skill_sigma', sigma=1)
            driver_skill = pm.Normal('driver_skill', 
                                   mu=driver_skill_mu, 
                                   sigma=driver_skill_sigma, 
                                   shape=len(self.driver_indices))
            
            # Team performance (hierarchical)
            team_perf_mu = pm.Normal('team_perf_mu', mu=0, sigma=1)
            team_perf_sigma = pm.HalfNormal('team_perf_sigma', sigma=1)
            team_perf = pm.Normal('team_perf', 
                                mu=team_perf_mu, 
                                sigma=team_perf_sigma, 
                                shape=len(self.team_indices))
            
            # Track effect if available
            track_effect = None
            if track_idx is not None:
                track_effect_mu = pm.Normal('track_effect_mu', mu=0, sigma=1)
                track_effect_sigma = pm.HalfNormal('track_effect_sigma', sigma=0.5)
                track_effect = pm.Normal('track_effect', 
                                       mu=track_effect_mu, 
                                       sigma=track_effect_sigma, 
                                       shape=len(self.track_indices))
            
            # Expected position
            mu = intercept + grid_effect * grid_position
            mu = mu + driver_skill[driver_idx] + team_perf[team_idx]
            
            if track_effect is not None and track_idx is not None:
                mu = mu + track_effect[track_idx]
            
            # Add noise term for variability in race outcomes
            sigma = pm.HalfNormal('sigma', sigma=2)
            
            # Likelihood (Normal since position can be continuous in the model)
            likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=position)
        
        return model
    
    def _build_gaussian_process_model(self, data: pd.DataFrame) -> pm.Model:
        """
        Build Gaussian Process Bayesian model for race predictions.
        
        Args:
            data: Training data
            
        Returns:
            PyMC3 model
        """
        # Prepare numerical features
        X = data[['GridPosition', 'QualifyingGapToPole']].values
        
        # Extract target
        position = data['Position'].values
        
        # Normalize features
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_normalized = (X - X_mean) / X_std
        
        # Build model
        with pm.Model() as model:
            # Length-scale for GP kernel
            ls = pm.HalfNormal('ls', sigma=2, shape=X.shape[1])
            
            # Signal standard deviation
            eta = pm.HalfNormal('eta', sigma=2)
            
            # Noise standard deviation
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            # GP covariance
            cov_func = eta**2 * pm.gp.cov.Matern52(X.shape[1], ls=ls)
            
            # GP mean function - linear model
            beta = pm.Normal('beta', 0, sigma=1, shape=X.shape[1])
            mean_func = pm.gp.mean.Linear(coeffs=beta)
            
            # GP prior
            gp = pm.gp.Marginal(cov_func=cov_func, mean_func=mean_func)
            
            # GP likelihood
            y_ = gp.marginal_likelihood('y', X=X_normalized, y=position, sigma=sigma)
        
        return model
    
    def train(self, training_data: pd.DataFrame) -> Dict:
        """
        Train Bayesian model on historical data.
        
        Args:
            training_data: Historical race data
            
        Returns:
            Training summary
        """
        # Build appropriate model
        if self.model_type == 'hierarchical':
            self.model = self._build_hierarchical_model(training_data)
        elif self.model_type == 'gp':
            self.model = self._build_gaussian_process_model(training_data)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Sample from posterior
        with self.model:
            self.trace = pm.sample(
                self.samples,
                tune=self.tune,
                chains=self.chains,
                return_inferencedata=True
            )
        
        # Compute model summary
        summary = az.summary(self.trace)
        
        return {
            'model_type': self.model_type,
            'samples': self.samples,
            'summary': summary
        }
    
    def predict(self, 
               quali_data: pd.DataFrame,
               samples: int = 500,
               return_samples: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]:
        """
        Make probabilistic predictions with uncertainty estimates.
        
        Args:
            quali_data: Qualifying data
            samples: Number of posterior samples to use
            return_samples: Whether to return raw position samples
            
        Returns:
            DataFrame with predicted positions and uncertainties,
            optionally with raw position samples
        """
        if self.model is None or self.trace is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare data for prediction
        if self.model_type == 'hierarchical':
            # Convert categorical variables to indices
            driver_indices = [
                self.driver_indices.get(d, -1) for d in quali_data['Driver']
            ]
            
            # Handle missing drivers by using the mean driver skill
            driver_skill_samples = self.trace.posterior['driver_skill'].values
            mean_driver_skill = driver_skill_samples.mean()
            
            # Make predictions with each posterior sample
            position_samples = []
            
            with self.model:
                # Extract posterior samples
                intercept_samples = self.trace.posterior['intercept'].values.flatten()
                grid_effect_samples = self.trace.posterior['grid_effect'].values.flatten()
                driver_skill_samples = self.trace.posterior['driver_skill'].values
                team_perf_samples = self.trace.posterior['team_perf'].values
                
                # Pick random subset of samples
                sample_indices = np.random.choice(len(intercept_samples), size=samples, replace=False)
                
                for i in range(len(quali_data)):
                    driver_idx = driver_indices[i]
                    
                    if driver_idx == -1:
                        # New driver, use mean skill
                        driver_skills = np.ones(samples) * mean_driver_skill
                    else:
                        # Extract driver skill samples
                        driver_skills = driver_skill_samples[:, :, driver_idx].flatten()[sample_indices]
                    
                    grid_pos = quali_data['GridPosition'].iloc[i]
                    
                    # Get team index
                    team = quali_data['Team'].iloc[i]
                    team_idx = self.team_indices.get(team, -1)
                    
                    if team_idx == -1:
                        # New team, use mean performance
                        team_perfs = np.ones(samples) * team_perf_samples.mean()
                    else:
                        # Extract team performance samples
                        team_perfs = team_perf_samples[:, :, team_idx].flatten()[sample_indices]
                    
                    # Compute position samples
                    pos_samples = (
                        intercept_samples[sample_indices] + 
                        grid_effect_samples[sample_indices] * grid_pos +
                        driver_skills +
                        team_perfs
                    )
                    
                    # Add noise
                    sigma_samples = self.trace.posterior['sigma'].values.flatten()[sample_indices]
                    pos_samples = pos_samples + np.random.normal(0, sigma_samples)
                    
                    position_samples.append(pos_samples)
            
            # Convert to array
            position_samples = np.array(position_samples)
        
        elif self.model_type == 'gp':
            # Prepare features
            X = quali_data[['GridPosition', 'QualifyingGapToPole']].values
            
            # Normalize features
            X_mean = X.mean(axis=0)
            X_std = X.std(axis=0)
            X_normalized = (X - X_mean) / X_std
            
            # Make predictions
            with self.model:
                gp = pm.gp.Marginal(
                    cov_func=self.model.cov_func,
                    mean_func=self.model.mean_func
                )
                
                # Sample from posterior predictive
                position_samples = gp.conditional(
                    'pred', X_normalized, 
                    samples=samples
                ).eval()
                
                # Transpose to match hierarchical model output
                position_samples = position_samples.T
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Calculate statistics from samples
        mean_positions = position_samples.mean(axis=1)
        std_positions = position_samples.std(axis=1)
        lower_positions = np.percentile(position_samples, 5, axis=1)
        upper_positions = np.percentile(position_samples, 95, axis=1)
        
        # Create results DataFrame
        results = quali_data.copy()
        results['PredictedPosition'] = mean_positions
        results['PositionStd'] = std_positions
        results['PositionLower'] = lower_positions
        results['PositionUpper'] = upper_positions
        
        # Sort by predicted position
        results = results.sort_values('PredictedPosition')
        
        # Assign final positions (1, 2, 3, etc.)
        results['Position'] = range(1, len(results) + 1)
        
        if return_samples:
            return results, position_samples
        else:
            return results
    
    def save(self, directory: str) -> str:
        """
        Save the trained model.
        
        Args:
            directory: Directory to save model
            
        Returns:
            Path to saved model
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save trace
        az.to_netcdf(self.trace, os.path.join(directory, 'trace.nc'))
        
        # Save indices and metadata
        metadata = {
            'model_type': self.model_type,
            'samples': self.samples,
            'tune': self.tune,
            'chains': self.chains,
            'driver_indices': self.driver_indices,
            'team_indices': self.team_indices,
            'track_indices': self.track_indices
        }
        
        with open(os.path.join(directory, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        return directory
    
    def load(self, directory: str) -> 'BayesianRacePredictionModel':
        """
        Load the trained model.
        
        Args:
            directory: Directory to load model from
            
        Returns:
            Loaded model
        """
        # Load metadata
        with open(os.path.join(directory, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        # Set attributes
        self.model_type = metadata['model_type']
        self.samples = metadata['samples']
        self.tune = metadata['tune']
        self.chains = metadata['chains']
        self.driver_indices = metadata['driver_indices']
        self.team_indices = metadata['team_indices']
        self.track_indices = metadata['track_indices']
        
        # Load trace
        self.trace = az.from_netcdf(os.path.join(directory, 'trace.nc'))
        
        # Rebuild model (simplified, would need full data to properly rebuild)
        # For prediction purposes, we don't need the full model, just the trace
        
        return self
    
    def plot_driver_skill_distribution(self, top_n: int = 10) -> plt.Figure:
        """
        Plot posterior distribution of driver skills.
        
        Args:
            top_n: Number of top drivers to show
            
        Returns:
            Matplotlib figure
        """
        if self.model_type != 'hierarchical':
            raise ValueError("Driver skill plot only available for hierarchical model")
        
        if self.trace is None:
            raise ValueError("Model not trained")
        
        # Extract driver skill samples
        driver_skills = self.trace.posterior['driver_skill'].mean(dim=['chain', 'draw']).values
        
        # Get top N drivers by skill
        driver_names = list(self.driver_indices.keys())
        driver_indices = list(self.driver_indices.values())
        
        driver_skill_df = pd.DataFrame({
            'Driver': driver_names,
            'Skill': driver_skills[driver_indices]
        })
        
        top_drivers = driver_skill_df.sort_values('Skill').tail(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot horizontal bars
        ax.barh(top_drivers['Driver'], top_drivers['Skill'])
        
        # Add title and labels
        ax.set_title('Estimated Driver Skill (Posterior Mean)', fontsize=14)
        ax.set_xlabel('Skill Level', fontsize=12)
        ax.set_ylabel('Driver', fontsize=12)
        
        # Add grid
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def plot_position_uncertainty(self, prediction_results: pd.DataFrame) -> plt.Figure:
        """
        Plot predicted positions with uncertainty intervals.
        
        Args:
            prediction_results: Prediction results from predict()
            
        Returns:
            Matplotlib figure
        """
        # Sort by predicted position
        sorted_results = prediction_results.sort_values('PredictedPosition')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot predicted positions with error bars
        ax.errorbar(
            x=range(len(sorted_results)),
            y=sorted_results['PredictedPosition'],
            yerr=[
                sorted_results['PredictedPosition'] - sorted_results['PositionLower'],
                sorted_results['PositionUpper'] - sorted_results['PredictedPosition']
            ],
            fmt='o',
            capsize=5,
            elinewidth=1,
            markeredgewidth=1
        )
        
        # Add driver labels
        for i, (_, row) in enumerate(sorted_results.iterrows()):
            ax.text(
                i, row['PredictedPosition'] - 0.5,
                row['Driver'],
                rotation=45,
                ha='right',
                va='top',
                fontsize=9
            )
        
        # Invert y-axis so 1st position is at the top
        ax.invert_yaxis()
        
        # Set title and labels
        ax.set_title('Predicted Positions with Uncertainty', fontsize=14)
        ax.set_xlabel('Driver Ranking', fontsize=12)
        ax.set_ylabel('Predicted Position', fontsize=12)
        ax.set_xticks([])  # Hide x-axis ticks
        
        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig