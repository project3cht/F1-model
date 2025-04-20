"""
Model visualization module for F1 race predictions.

This module provides functionality to visualize model training, 
detect overfitting, and analyze model performance.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import joblib
import os
from matplotlib.gridspec import GridSpec
import logging

class ModelTrainingVisualizer:
    """
    Visualizer for model training process, with focus on detecting overfitting
    and optimizing model performance.
    """
    
    def __init__(self, name="TrainingVisualizer", output_dir="visualizations"):
        """
        Initialize the training visualizer.
        
        Args:
            name (str): Name of the visualizer
            output_dir (str): Directory to save visualization plots
        """
        self.name = name
        self.output_dir = output_dir
        self.training_history = {}
        self.logger = logging.getLogger(f'f1_prediction.{name}')
        
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            self.logger.info(f"Created visualization directory: {output_dir}")
    
    def visualize_learning_curve(self, estimator, X, y, cv=5, n_jobs=-1, 
                                train_sizes=np.linspace(0.1, 1.0, 10),
                                title="Learning Curve"):
        """
        Plot learning curve to detect overfitting.
        
        Args:
            estimator: Estimator object (unfitted)
            X: Feature matrix
            y: Target vector
            cv: Cross-validation folds
            n_jobs: Number of jobs for parallel execution
            train_sizes: Array of training sizes to evaluate
            title: Title for the plot
            
        Returns:
            fig: Matplotlib figure
        """
        self.logger.info(f"Generating learning curve for {estimator.__class__.__name__}")
        
        # Calculate learning curve
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, 
            train_sizes=train_sizes, scoring='neg_mean_absolute_error'
        )
        
        # Convert negative MAE to positive for easier interpretation
        train_scores = -train_scores
        val_scores = -val_scores
        
        # Calculate mean and std for train and validation scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot learning curve
        ax.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
        ax.plot(train_sizes, val_mean, 'o-', color='g', label='Cross-validation score')
        
        # Plot standard deviation bands
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                      alpha=0.1, color='r')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                      alpha=0.1, color='g')
        
        # Add overfitting detection
        # If validation score is significantly worse than training at high sample sizes
        if val_mean[-1] > train_mean[-1] * 1.2:  # 20% difference threshold
            ax.axvspan(train_sizes[-3], train_sizes[-1], alpha=0.2, color='yellow')
            ax.text(train_sizes[-2], max(train_mean.max(), val_mean.max()) * 0.9,
                   "Potential Overfitting Zone", 
                   ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", fc='yellow', alpha=0.3))
            self.logger.warning("Detected potential overfitting in learning curve")
        
        # Add labels and title
        ax.set_xlabel('Training Examples')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True)
        
        # Save figure
        filename = f"{title.replace(' ', '_').lower()}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved learning curve to {filepath}")
        
        return fig
    
    def visualize_feature_importance(self, model, feature_names, title="Feature Importance"):
        """
        Visualize feature importance for the trained model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            title: Title for the plot
            
        Returns:
            fig: Matplotlib figure
        """
        self.logger.info(f"Generating feature importance visualization for {model.__class__.__name__}")
        
        # Check if model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            # Get feature importances directly from model
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            importances = np.abs(model.coef_)
            if importances.ndim > 1:
                importances = importances.mean(axis=0)
        else:
            # For other models (like pipelines), check final estimator
            if hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'feature_importances_'):
                importances = model.steps[-1][1].feature_importances_
            elif hasattr(model, 'final_estimator_') and hasattr(model.final_estimator_, 'feature_importances_'):
                importances = model.final_estimator_.feature_importances_
            else:
                # Try permutation importance as fallback
                try:
                    self.logger.info("Attempting to use permutation importance")
                    X_sample = getattr(model, '_last_X', None)
                    y_sample = getattr(model, '_last_y', None)
                    
                    if X_sample is not None and y_sample is not None:
                        perm_importance = permutation_importance(model, X_sample, y_sample, n_repeats=10)
                        importances = perm_importance.importances_mean
                    else:
                        raise ValueError("Model doesn't have accessible feature importances and no sample data available")
                except Exception as e:
                    self.logger.error(f"Could not determine feature importance: {e}")
                    raise ValueError(f"Model doesn't have accessible feature importances: {e}")
        
        # Ensure we don't have more features than feature names
        if len(importances) > len(feature_names):
            importances = importances[:len(feature_names)]
        elif len(importances) < len(feature_names):
            feature_names = feature_names[:len(importances)]
            
        # Create DataFrame with features and importances
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot horizontal bar chart
        sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
        
        # Add labels and title
        ax.set_title(title)
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        
        # Save figure
        filename = f"{title.replace(' ', '_').lower()}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved feature importance to {filepath}")
        
        return fig, feature_importance
    
    def visualize_validation_curve(self, estimator, X, y, param_name, param_range, 
                                  cv=5, scoring='neg_mean_absolute_error',
                                  title="Validation Curve"):
        """
        Plot validation curve for hyperparameter tuning.
        
        Args:
            estimator: Estimator object (unfitted)
            X: Feature matrix
            y: Target vector
            param_name: Name of parameter to vary
            param_range: Range of values for the parameter
            cv: Cross-validation folds
            scoring: Scoring metric
            title: Title for the plot
            
        Returns:
            fig: Matplotlib figure
            optimal_param: Optimal parameter value
        """
        self.logger.info(f"Generating validation curve for parameter '{param_name}'")
        
        # Calculate validation curve
        train_scores, val_scores = validation_curve(
            estimator, X, y, param_name=param_name, param_range=param_range,
            cv=cv, scoring=scoring, n_jobs=-1
        )
        
        # If using negative scoring metric, convert to positive
        if scoring.startswith('neg_'):
            train_scores = -train_scores
            val_scores = -val_scores
        
        # Calculate mean and std for train and validation scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot validation curve
        ax.plot(param_range, train_mean, 'o-', color='r', label='Training score')
        ax.plot(param_range, val_mean, 'o-', color='g', label='Cross-validation score')
        
        # Plot standard deviation bands
        ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, 
                      alpha=0.1, color='r')
        ax.fill_between(param_range, val_mean - val_std, val_mean + val_std, 
                      alpha=0.1, color='g')
        
        # Detect optimal parameter value (lowest validation error)
        best_idx = np.argmin(val_mean)
        best_param = param_range[best_idx]
        best_score = val_mean[best_idx]
        
        # Mark optimal parameter
        ax.axvline(x=best_param, color='blue', linestyle='--', alpha=0.8)
        ax.text(best_param, min(train_mean.min(), val_mean.min()) * 0.9,
               f"Optimal {param_name}={best_param:.4g}\nScore={best_score:.4g}", 
               ha='center', va='bottom',
               bbox=dict(boxstyle="round,pad=0.3", fc='blue', alpha=0.1))
        
        # Add labels and title
        ax.set_xlabel(param_name)
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True)
        
        # Check if x-axis should be log scale
        if max(param_range) / min(param_range) > 100:
            ax.set_xscale('log')
        
        # Save figure
        filename = f"validation_curve_{param_name.replace('.', '_')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved validation curve to {filepath}")
        
        self.logger.info(f"Optimal {param_name} = {best_param:.6g} (Score: {best_score:.6g})")
        
        return fig, best_param
    
    def track_training_progress(self, epoch, train_metrics, val_metrics, model_name):
        """
        Track and log training progress for a model.
        
        Args:
            epoch (int): Current training epoch
            train_metrics (dict): Training metrics
            val_metrics (dict): Validation metrics
            model_name (str): Name of the model
            
        Returns:
            bool: True if improvement detected, False otherwise
        """
        # Initialize history for this model if it doesn't exist
        if model_name not in self.training_history:
            self.training_history[model_name] = {
                'epoch': [],
                'train_mae': [],
                'val_mae': [],
                'train_rmse': [],
                'val_rmse': [],
                'best_val_mae': float('inf'),
                'best_epoch': 0,
                'no_improve_count': 0
            }
        
        history = self.training_history[model_name]
        
        # Store current metrics
        history['epoch'].append(epoch)
        history['train_mae'].append(train_metrics['mae'])
        history['val_mae'].append(val_metrics['mae'])
        history['train_rmse'].append(train_metrics['rmse'])
        history['val_rmse'].append(val_metrics['rmse'])
        
        # Check for improvement
        improvement = False
        if val_metrics['mae'] < history['best_val_mae']:
            improvement = True
            history['best_val_mae'] = val_metrics['mae']
            history['best_epoch'] = epoch
            history['no_improve_count'] = 0
            self.logger.info(f"New best {model_name} at epoch {epoch}: MAE={val_metrics['mae']:.4f}, RMSE={val_metrics['rmse']:.4f}")
        else:
            history['no_improve_count'] += 1
            self.logger.info(f"{model_name} epoch {epoch}: MAE={val_metrics['mae']:.4f} (no improvement for {history['no_improve_count']} checks)")
        
        # Check for early stopping
        should_stop = history['no_improve_count'] >= 5  # No improvement for 5 consecutive checks
        if should_stop:
            self.logger.info(f"Early stopping {model_name} training at epoch {epoch}")
        
        return improvement, should_stop
    
    def plot_training_history(self, model_name, title=None):
        """
        Plot training history metrics.
        
        Args:
            model_name: Name of the model
            title: Title for the plot
            
        Returns:
            fig: Matplotlib figure
        """
        if model_name not in self.training_history:
            raise ValueError(f"No training history found for {model_name}")
        
        history = self.training_history[model_name]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot MAE
        ax1.plot(history['epoch'], history['train_mae'], 'b-', label='Training MAE')
        ax1.plot(history['epoch'], history['val_mae'], 'r-', label='Validation MAE')
        ax1.set_ylabel('Mean Absolute Error')
        ax1.set_title(f"{title or model_name} - MAE")
        ax1.legend()
        ax1.grid(True)
        
        # Plot RMSE
        ax2.plot(history['epoch'], history['train_rmse'], 'b-', label='Training RMSE')
        ax2.plot(history['epoch'], history['val_rmse'], 'r-', label='Validation RMSE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Root Mean Squared Error')
        ax2.set_title(f"{title or model_name} - RMSE")
        ax2.legend()
        ax2.grid(True)
        
        # Mark best epoch
        best_epoch = history['best_epoch']
        for ax in [ax1, ax2]:
            ax.axvline(x=best_epoch, color='green', linestyle='--')
            ax.text(best_epoch, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1,
                   f"Best Epoch: {best_epoch}",
                   ha='right', va='bottom',
                   bbox=dict(boxstyle="round,pad=0.3", fc='green', alpha=0.1))
        
        # Detect potential overfitting
        epochs = history['epoch']
        train_mae = history['train_mae']
        val_mae = history['val_mae']
        
        # Find points where validation error increases while training error decreases
        for i in range(2, len(epochs)):
            if (val_mae[i] > val_mae[i-1] and train_mae[i] < train_mae[i-1]):
                # Mark potential overfitting
                start_epoch = epochs[i-1]
                ax1.axvspan(start_epoch, max(epochs), alpha=0.2, color='yellow')
                ax2.axvspan(start_epoch, max(epochs), alpha=0.2, color='yellow')
                ax1.text(start_epoch + (max(epochs) - start_epoch)/2, 
                       max(train_mae + val_mae) * 0.9,
                       "Potential Overfitting",
                       ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.3", fc='yellow', alpha=0.3))
                
                self.logger.warning(f"Potential overfitting detected for {model_name} starting at epoch {start_epoch}")
                break
        
        plt.tight_layout()
        
        # Save figure
        filename = f"training_history_{model_name.lower().replace(' ', '_')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved training history to {filepath}")
        
        return fig
    
    def create_model_dashboard(self, model, X_train, y_train, X_val, y_val, feature_names,
                              params_to_vary=None, model_name="Model"):
        """
        Create a comprehensive model dashboard with multiple visualizations.
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            feature_names: List of feature names
            params_to_vary: Dict of parameter names and ranges for validation curves
            model_name: Name of the model for display
            
        Returns:
            fig: Matplotlib figure with dashboard
        """
        self.logger.info(f"Creating model dashboard for {model_name}")
        
        # Create figure with grid layout
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 2, figure=fig)
        
        # Save X and y for future reference (used in feature importance)
        setattr(model, '_last_X', X_val)
        setattr(model, '_last_y', y_val)
        
        # 1. Learning curve (top left)
        self.logger.info("Generating learning curve for dashboard")
        ax1 = fig.add_subplot(gs[0, 0])
        try:
            # Use a clone of the model to avoid modifying the original
            clone_estimator = joblib.clone(model)
            train_sizes, train_scores, val_scores = learning_curve(
                clone_estimator, np.vstack((X_train, X_val)), 
                np.concatenate((y_train, y_val)),
                cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='neg_mean_absolute_error'
            )
            train_scores = -train_scores
            val_scores = -val_scores
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            ax1.plot(train_sizes, train_mean, 'o-', color='r', label='Training')
            ax1.plot(train_sizes, val_mean, 'o-', color='g', label='Cross-validation')
            ax1.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
            ax1.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='g')
            ax1.set_xlabel('Training Examples')
            ax1.set_ylabel('Mean Absolute Error')
            ax1.set_title('Learning Curve')
            ax1.legend(loc='best')
            ax1.grid(True)
            
            # Add overfitting detection
            if val_mean[-1] > train_mean[-1] * 1.2:  # 20% difference threshold
                ax1.axvspan(train_sizes[-3], train_sizes[-1], alpha=0.2, color='yellow')
                ax1.text(train_sizes[-2], max(train_mean.max(), val_mean.max()) * 0.9,
                       "Potential Overfitting Zone", 
                       ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.3", fc='yellow', alpha=0.3))
        except Exception as e:
            self.logger.error(f"Could not generate learning curve: {e}")
            ax1.text(0.5, 0.5, f"Could not generate learning curve:\n{str(e)}",
                   ha='center', va='center', fontsize=10)
            ax1.set_title('Learning Curve (Not Available)')
        
        # 2. Feature importance (top right)
        self.logger.info("Generating feature importance for dashboard")
        ax2 = fig.add_subplot(gs[0, 1])
        try:
            # Try different methods to get feature importances
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
                if importances.ndim > 1:
                    importances = importances.mean(axis=0)
            elif hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'feature_importances_'):
                importances = model.steps[-1][1].feature_importances_
            elif hasattr(model, 'final_estimator_') and hasattr(model.final_estimator_, 'feature_importances_'):
                importances = model.final_estimator_.feature_importances_
            else:
                # Use permutation importance as fallback
                perm_importance = permutation_importance(model, X_val, y_val, n_repeats=10)
                importances = perm_importance.importances_mean
            
            # Ensure we don't have more features than feature names
            if len(importances) > len(feature_names):
                importances = importances[:len(feature_names)]
            elif len(importances) < len(feature_names):
                feature_names = feature_names[:len(importances)]
                
            # Create DataFrame with features and importances
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
            feature_importance = feature_importance.sort_values('Importance', ascending=False)
            
            # Plot top 15 features
            top_n = min(15, len(feature_importance))
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(top_n), ax=ax2)
            ax2.set_title('Feature Importance')
            ax2.set_xlabel('Importance')
            ax2.set_ylabel('Feature')
        except Exception as e:
            self.logger.error(f"Could not compute feature importance: {e}")
            ax2.text(0.5, 0.5, f"Could not compute feature importance:\n{str(e)}",
                   ha='center', va='center', fontsize=10)
            ax2.set_title('Feature Importance (Not Available)')
        
        # 3. Training history (middle row)
        ax3 = fig.add_subplot(gs[1, :])
        
        if model_name in self.training_history:
            self.logger.info("Plotting training history for dashboard")
            history = self.training_history[model_name]
            epochs = history['epoch']
            
            ax3.plot(epochs, history['train_mae'], 'b-', label='Training MAE')
            ax3.plot(epochs, history['val_mae'], 'r-', label='Validation MAE')
            
            # Find best epoch
            best_epoch = history['best_epoch']
            ax3.axvline(x=best_epoch, color='green', linestyle='--')
            ax3.text(best_epoch, min(history['train_mae']) * 0.9,
                   f"Best Epoch: {best_epoch}",
                   ha='right', va='bottom',
                   bbox=dict(boxstyle="round,pad=0.3", fc='green', alpha=0.1))
            
            # Check for overfitting
            for i in range(2, len(epochs)):
                if (history['val_mae'][i] > history['val_mae'][i-1] and 
                    history['train_mae'][i] < history['train_mae'][i-1]):
                    ax3.axvspan(epochs[i-1], max(epochs), alpha=0.2, color='yellow')
                    ax3.text(epochs[i-1] + (max(epochs) - epochs[i-1])/2, 
                           max(history['train_mae'] + history['val_mae']) * 0.9,
                           "Potential Overfitting",
                           ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.3", fc='yellow', alpha=0.3))
                    break
            
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Mean Absolute Error')
            ax3.set_title('Training History')
            ax3.legend(loc='best')
            ax3.grid(True)
        else:
            ax3.text(0.5, 0.5, "No training history available",
                   ha='center', va='center', fontsize=14)
            ax3.set_title('Training History (Not Available)')
        
        # 4 & 5. Bottom row: Validation curves for parameters or predictions
        if params_to_vary and len(params_to_vary) > 0:
            self.logger.info("Generating validation curves for dashboard")
            for i, (param_name, param_range) in enumerate(params_to_vary.items()):
                if i >= 2:  # Only show up to 2 parameter validation curves
                    break
                
                ax = fig.add_subplot(gs[2, i])
                try:
                    # Use a clone of the model to avoid modifying the original
                    clone_estimator = joblib.clone(model)
                    train_scores, val_scores = validation_curve(
                        clone_estimator, X_train, y_train, param_name=param_name, 
                        param_range=param_range, cv=5, scoring='neg_mean_absolute_error'
                    )
                    train_scores = -train_scores
                    val_scores = -val_scores
                    
                    train_mean = np.mean(train_scores, axis=1)
                    train_std = np.std(train_scores, axis=1)
                    val_mean = np.mean(val_scores, axis=1)
                    val_std = np.std(val_scores, axis=1)
                    
                    ax.plot(param_range, train_mean, 'o-', color='r', label='Training')
                    ax.plot(param_range, val_mean, 'o-', color='g', label='Cross-validation')
                    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
                    ax.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='g')
                    
                    # Find optimal parameter value
                    best_idx = np.argmin(val_mean)
                    best_param = param_range[best_idx]
                    ax.axvline(x=best_param, color='blue', linestyle='--')
                    
                    ax.set_xlabel(param_name)
                    ax.set_ylabel('Mean Absolute Error')
                    ax.set_title(f'Validation Curve - {param_name}')
                    ax.legend(loc='best')
                    ax.grid(True)
                    
                    # Use log scale if appropriate
                    if max(param_range) / min(param_range) > 100:
                        ax.set_xscale('log')
                except Exception as e:
                    self.logger.error(f"Could not compute validation curve for {param_name}: {e}")
                    ax.text(0.5, 0.5, f"Could not compute validation curve:\n{str(e)}",
                          ha='center', va='center', fontsize=10)
                    ax.set_title(f'Validation Curve - {param_name} (Not Available)')
        else:
            # If no parameters to vary, show predictions vs actual
            self.logger.info("Generating prediction scatter plot for dashboard")
            ax4 = fig.add_subplot(gs[2, 0])
            y_pred = model.predict(X_val)
            ax4.scatter(y_val, y_pred, alpha=0.5)
            ax4.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--')
            ax4.set_xlabel('Actual')
            ax4.set_ylabel('Predicted')
            ax4.set_title('Predictions vs Actual (Validation Set)')
            
            # Add error metrics to the plot
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            
            ax4.text(0.05, 0.95, 
                   f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}",
                   transform=ax4.transAxes, fontsize=12,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Calculate error distribution
            self.logger.info("Generating error distribution plot for dashboard")
            ax5 = fig.add_subplot(gs[2, 1])
            errors = y_pred - y_val
            sns.histplot(errors, kde=True, ax=ax5)
            ax5.set_xlabel('Prediction Error')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Error Distribution')
            
            # Add error statistics
            error_mean = np.mean(errors)
            error_std = np.std(errors)
            ax5.text(0.05, 0.95,
                   f"Mean Error: {error_mean:.4f}\nStd Dev: {error_std:.4f}",
                   transform=ax5.transAxes, fontsize=12,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add main title
        plt.suptitle(f'Model Dashboard - {model_name}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save figure
        filename = f"model_dashboard_{model_name.lower().replace(' ', '_')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved model dashboard to {filepath}")
        
        return fig
    
    def visualize_model_comparison(self, models_dict, X_val, y_val, title="Model Comparison"):
        """
        Compare multiple models' performance.
        
        Args:
            models_dict: Dictionary of {model_name: model_object}
            X_val: Validation features
            y_val: Validation targets
            title: Title for the plot
            
        Returns:
            fig: Matplotlib figure
        """
        self.logger.info(f"Comparing performance of {len(models_dict)} models")
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Store metrics
        model_names = []
        mae_scores = []
        rmse_scores = []
        
        # Evaluate each model
        for name, model in models_dict.items():
            self.logger.info(f"Evaluating model: {name}")
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            # Store results
            model_names.append(name)
            mae_scores.append(mae)
            rmse_scores.append(rmse)
            
            self.logger.info(f"Model {name}: MAE={mae:.4f}, RMSE={rmse:.4f}")
        
        # Create DataFrames for plotting
        results_df = pd.DataFrame({
            'Model': model_names,
            'MAE': mae_scores,
            'RMSE': rmse_scores
        })
        
        # Plot MAE
        sns.barplot(x='Model', y='MAE', data=results_df, ax=ax1)
        ax1.set_title('Model Comparison - MAE (lower is better)')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Mean Absolute Error')
        
        # Annotate bars with values
        for i, v in enumerate(mae_scores):
            ax1.text(i, v + 0.05, f"{v:.4f}", ha='center')
        
        # Plot RMSE
        sns.barplot(x='Model', y='RMSE', data=results_df, ax=ax2)
        ax2.set_title('Model Comparison - RMSE (lower is better)')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Root Mean Squared Error')
        
        # Annotate bars with values
        for i, v in enumerate(rmse_scores):
            ax2.text(i, v + 0.05, f"{v:.4f}", ha='center')
        
        # Add main title
        plt.suptitle(title, fontsize=14)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        filename = f"model_comparison_{title.lower().replace(' ', '_')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved model comparison to {filepath}")
        
        return fig
    
    def visualize_residuals(self, model, X, y, title="Residual Analysis"):
        """
        Create visualization of residuals for model diagnostics.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            title: Title for the plot
            
        Returns:
            fig: Matplotlib figure
        """
        self.logger.info(f"Performing residual analysis for {model.__class__.__name__}")
        
        # Predict values
        y_pred = model.predict(X)
        
        # Calculate residuals
        residuals = y - y_pred
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.5)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted Values')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add LOWESS smoothed line if available
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            z = lowess(residuals, y_pred, frac=0.3)
            ax1.plot(z[:, 0], z[:, 1], 'r-', linewidth=2)
        except ImportError:
            self.logger.info("statsmodels not available for LOWESS smoothing")
        
        # 2. Histogram of Residuals
        sns.histplot(residuals, kde=True, ax=ax2)
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Residuals')
        
        # 3. Q-Q Plot (Check for normality)
        from scipy import stats
        import matplotlib.pyplot as plt
        
        # Calculate theoretical quantiles
        sorted_residuals = np.sort(residuals)
        norm_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))
        
        # Create Q-Q plot
        ax3.scatter(norm_quantiles, sorted_residuals, alpha=0.5)
        ax3.plot([np.min(norm_quantiles), np.max(norm_quantiles)],
               [np.min(norm_quantiles), np.max(norm_quantiles)],
               'r--')
        ax3.set_xlabel('Theoretical Quantiles')
        ax3.set_ylabel('Sample Quantiles')
        ax3.set_title('Q-Q Plot (Normality Check)')
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # 4. Residuals vs Actual
        ax4.scatter(y, residuals, alpha=0.5)
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_xlabel('Actual Values')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Residuals vs Actual Values')
        ax4.grid(True, linestyle='--', alpha=0.7)
        
        # Calculate residual statistics
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        
        # Add statistics as text
        stats_text = (f"Residual Statistics:\n"
                     f"Mean: {residual_mean:.4f}\n"
                     f"Std Dev: {residual_std:.4f}\n"
                     f"Min: {np.min(residuals):.4f}\n"
                     f"Max: {np.max(residuals):.4f}")
        
        fig.text(0.02, 0.02, stats_text, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add main title
        plt.suptitle(title, fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure
        filename = f"residual_analysis_{title.lower().replace(' ', '_')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved residual analysis to {filepath}")
        
        return fig