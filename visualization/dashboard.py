"""
Interactive dashboard visualizations for F1 race predictions.

This module provides functionality for creating interactive dashboards
and comprehensive race reports.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from utils.constants import TEAM_COLORS, ROOKIES
from utils.helpers import ensure_directory

def ensure_visualization_folder():
    """
    Ensure the visualization output folder exists.
    
    Returns:
        str: Path to the visualization folder
    """
    return ensure_directory('visualizations')

def create_interactive_dashboard(predictions, track_name=None, tire_data=None, 
                                save_path=None, additional_data=None):
    """
    Create a comprehensive dashboard visualization for race predictions.
    
    Args:
        predictions (DataFrame): Predicted race results
        track_name (str, optional): Name of the track
        tire_data (dict, optional): Tire-specific data
        save_path (str, optional): Path to save the visualization
        additional_data (dict, optional): Additional data for the dashboard
        
    Returns:
        str: Path to the saved dashboard
    """
    # Set up figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 3, figure=fig)
    
    # 1. Header - Race Prediction Title
    header_ax = fig.add_subplot(gs[0, :])
    header_ax.axis('off')
    
    # Create header text
    header_title = "F1 Race Prediction Dashboard"
    if track_name:
        header_title += f" - {track_name}"
    
    header_ax.text(0.5, 0.6, header_title, ha='center', va='center', fontsize=24, fontweight='bold')
    
    # Add race info
    if additional_data and 'race_info' in additional_data:
        race_info = additional_data['race_info']
        info_text = f"Round: {race_info.get('round', 'N/A')} | "
        info_text += f"Date: {race_info.get('date', 'N/A')} | "
        info_text += f"Distance: {race_info.get('distance', 'N/A')} km | "
        info_text += f"Laps: {race_info.get('laps', 'N/A')}"
        
        header_ax.text(0.5, 0.3, info_text, ha='center', va='center', fontsize=14)
    
    # 2. Top 10 Finishers
    top10_ax = fig.add_subplot(gs[1, 0])
    
    # Get top 10 predictions
    top_10 = predictions.head(10).copy()
    
    # Create bar chart
    bars = top10_ax.barh(top_10['Driver'], top_10['Interval (s)'], color='skyblue')
    
    # Color bars by team
    for i, bar in enumerate(bars):
        team = top_10.iloc[i]['Team']
        color = TEAM_COLORS.get(team, 'gray')
        bar.set_color(color)
    
    # Add interval values
    for i, value in enumerate(top_10['Interval']):
        if value != "WINNER":
            top10_ax.text(top_10.iloc[i]['Interval (s)'] + 0.1, i, value, va='center')
        else:
            top10_ax.text(0.1, i, value, va='center', weight='bold')
    
    top10_ax.set_title('Predicted Top 10 Finishers', fontsize=14)
    top10_ax.set_xlabel('Time Gap to Winner (seconds)')
    top10_ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 3. Grid vs. Finish
    grid_ax = fig.add_subplot(gs[1, 1])
    
    # Create scatter plot
    for i, row in predictions.iterrows():
        team = row['Team']
        color = TEAM_COLORS.get(team, 'gray')
        grid_ax.scatter(
            row['Grid'], 
            row['Position'], 
            color=color, 
            s=100, 
            label=team if team not in grid_ax.get_legend_handles_labels()[1] else ""
        )
        grid_ax.text(row['Grid'] + 0.1, row['Position'] + 0.1, row['Driver'], fontsize=8)
    
    # Add diagonal line (no position change)
    max_pos = max(predictions['Grid'].max(), predictions['Position'].max())
    grid_ax.plot([1, max_pos], [1, max_pos], 'k--', alpha=0.3)
    
    # Customize plot
    grid_ax.set_title('Grid vs. Predicted Finishing Positions', fontsize=14)
    grid_ax.set_xlabel('Grid Position')
    grid_ax.set_ylabel('Predicted Finishing Position')
    grid_ax.grid(True, linestyle='--', alpha=0.7)
    grid_ax.invert_yaxis()  # Invert y-axis so that 1st is at the top
    grid_ax.invert_xaxis()  # Invert x-axis so that 1st is at the left
    
    # 4. Team Performance
    team_ax = fig.add_subplot(gs[1, 2])
    
    # Group data by team
    team_performance = predictions.groupby('Team')['Position'].agg(['min', 'max', 'mean']).reset_index()
    team_performance = team_performance.sort_values('min')
    
    for i, row in team_performance.iterrows():
        team = row['Team']
        pos_min = row['min']
        pos_max = row['max']
        pos_avg = row['mean']
        
        # Get team color
        color = TEAM_COLORS.get(team, 'gray')
        
        # Plot range
        team_ax.plot([team, team], [pos_min, pos_max], 'o-', color=color, markersize=8, label=team)
        
        # Add average marker
        team_ax.plot(team, pos_avg, 'D', color='black', markersize=6)
        
        # Add text annotations
        team_ax.text(i, pos_min - 0.4, f"P{int(pos_min)}", ha='center', fontsize=8)
    
    team_ax.set_title('Team Performance', fontsize=14)
    team_ax.set_xlabel('Team')
    team_ax.set_ylabel('Position')
    team_ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    team_ax.set_yticks(range(1, 21))
    team_ax.invert_yaxis()  # Invert y-axis so position 1 is at the top
    team_ax.set_xticks(range(len(team_performance)))
    team_ax.set_xticklabels(team_performance['Team'], rotation=45, ha='right')
    
    # 5. Rookies Performance
    rookie_ax = fig.add_subplot(gs[2, 0])
    
    # Filter rookie drivers
    rookie_data = predictions[predictions['Driver'].isin(ROOKIES)].copy()
    
    if len(rookie_data) > 0:
        # Sort by position
        rookie_data = rookie_data.sort_values('Position')
        
        # Create bar chart for rookies
        bars = rookie_ax.barh(rookie_data['Driver'], rookie_data['Position'], color='skyblue')
        
        # Color bars by team
        for i, bar in enumerate(bars):
            team = rookie_data.iloc[i]['Team']
            color = TEAM_COLORS.get(team, 'gray')
            bar.set_color(color)
        
        # Add grid position for comparison
        for i, row in rookie_data.iterrows():
            rookie_ax.text(
                row['Position'] + 0.2, 
                i, 
                f"Grid: P{int(row['Grid'])}", 
                va='center', 
                fontsize=9
            )
        
        # Customize plot
        rookie_ax.set_title('Rookie Driver Performance', fontsize=14)
        rookie_ax.set_xlabel('Predicted Position')
        rookie_ax.grid(axis='x', linestyle='--', alpha=0.7)
        rookie_ax.invert_xaxis()  # Invert x-axis so position 1 is at the right
    else:
        rookie_ax.text(0.5, 0.5, "No rookie drivers in this race", ha='center', va='center', fontsize=12)
        rookie_ax.axis('off')
    
    # 6. Tire Strategy Insights
    tire_ax = fig.add_subplot(gs[2, 1:])
    
    if tire_data and 'optimal_stints' in tire_data:
        # Extract optimal stint data
        optimal_stints = tire_data['optimal_stints']
        
        if optimal_stints:
            # Create bar chart for optimal stint lengths
            compounds = [f"{x['display_name']} ({x['compound']})" for x in optimal_stints]
            stint_lengths = [x['optimal_stint'] for x in optimal_stints]
            
            bars = tire_ax.barh(compounds, stint_lengths)
            
            # Color bars based on compound
            for i, bar in enumerate(bars):
                display_name = optimal_stints[i]['display_name']
                
                if display_name == 'SOFT':
                    color = '#FF0000'  # Red
                elif display_name == 'MEDIUM':
                    color = '#FFFF00'  # Yellow
                else:  # HARD
                    color = '#CCCCCC'  # Light gray
                    
                bar.set_color(color)
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                tire_ax.text(
                    width + 0.5, 
                    bar.get_y() + bar.get_height()/2, 
                    f"{int(width)} laps", 
                    ha='left', 
                    va='center'
                )
            
            # Customize plot
            tire_ax.set_title('Optimal Tire Stint Lengths', fontsize=14)
            tire_ax.set_xlabel('Number of Laps')
            tire_ax.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Add strategy suggestions
            if 'crossovers' in tire_data:
                crossovers = tire_data['crossovers']
                
                crossover_text = "Tire Crossover Points:\n"
                for key, data in crossovers.items():
                    crossover_text += f"• {data['name1']} ↔ {data['name2']}: Lap {data['lap']:.1f}\n"
                
                # Add text box with crossover info
                props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                tire_ax.text(
                    0.05, 0.05, crossover_text, 
                    transform=tire_ax.transAxes,
                    fontsize=10,
                    verticalalignment='bottom',
                    bbox=props
                )
        else:
            tire_ax.text(0.5, 0.5, "No tire data available for strategy insights", ha='center', va='center', fontsize=12)
            tire_ax.axis('off')
    else:
        tire_ax.text(0.5, 0.5, "No tire data available for strategy insights", ha='center', va='center', fontsize=12)
        tire_ax.axis('off')
    
    plt.tight_layout()
    
    # Save dashboard
    if save_path:
        plt.savefig(save_path)
        print(f"Dashboard saved to {save_path}")
        saved_path = save_path
    else:
        # Use default path
        output_dir = ensure_visualization_folder()
        filename = f"{'race_dashboard' if not track_name else track_name.lower().replace(' ', '_') + '_dashboard'}.png"
        saved_path = os.path.join(output_dir, filename)
        plt.savefig(saved_path)
        print(f"Dashboard saved to {saved_path}")
    
    plt.close()
    
    return saved_path

def create_race_report(predictions, track_name=None, tire_data=None, 
                      monte_carlo_data=None, historical_data=None,
                      save_path=None):
    """
    Create a comprehensive race report with visualizations and analysis.
    
    Args:
        predictions (DataFrame): Predicted race results
        track_name (str, optional): Name of the track
        tire_data (dict, optional): Tire-specific data
        monte_carlo_data (dict, optional): Monte Carlo simulation data
        historical_data (dict, optional): Historical accuracy data
        save_path (str, optional): Path to save the report
        
    Returns:
        str: Path to the saved report
    """
    # Create output directory
    output_dir = ensure_visualization_folder()
    if save_path is None:
        save_path = os.path.join(output_dir, f"{'race_report' if not track_name else track_name.lower().replace(' ', '_') + '_report'}.html")
    
    # Create HTML report
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>F1 Race Prediction Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            .header { text-align: center; margin-bottom: 30px; background-color: #e10600; color: white; padding: 20px; }
            .section { margin-bottom: 40px; }
            h1, h2, h3 { font-weight: normal; }
            .race-summary { display: flex; justify-content: space-around; flex-wrap: wrap; }
            .summary-card { flex: 1; margin: 10px; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); min-width: 250px; max-width: 300px; }
            .results-table { width: 100%; border-collapse: collapse; }
            .results-table th, .results-table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            .results-table th { background-color: #f2f2f2; }
            .results-table tr:hover { background-color: #f5f5f5; }
            .visualization { margin: 20px 0; text-align: center; }
            .visualization img { max-width: 100%; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .team-color { display: inline-block; width: 12px; height: 12px; margin-right: 5px; border-radius: 50%; }
            .footnote { font-size: 12px; color: #666; text-align: center; margin-top: 50px; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>F1 Race Prediction Report</h1>
    """
    
    # Add track name if available
    if track_name:
        html += f"<h2>{track_name}</h2>"
    
    html += """
        </div>
        <div class="container">
            <div class="section">
                <h2>Race Prediction Summary</h2>
                <div class="race-summary">
    """
    
    # Calculate summary statistics
    winner = predictions.iloc[0]
    podium = predictions.head(3)
    points_positions = predictions.head(10)
    
    # Create summary cards
    html += f"""
                    <div class="summary-card">
                        <h3>Winner</h3>
                        <div><span class="team-color" style="background-color: {TEAM_COLORS.get(winner['Team'], '#999999')};"></span> {winner['Driver']}</div>
                        <div><strong>Team:</strong> {winner['Team']}</div>
                        <div><strong>Grid:</strong> P{int(winner['Grid'])}</div>
                    </div>
                    
                    <div class="summary-card">
                        <h3>Podium</h3>
                        <div>P1: {podium.iloc[0]['Driver']} ({podium.iloc[0]['Team']})</div>
                        <div>P2: {podium.iloc[1]['Driver']} ({podium.iloc[1]['Team']})</div>
                        <div>P3: {podium.iloc[2]['Driver']} ({podium.iloc[2]['Team']})</div>
                    </div>
    """
    
    # Add team with most cars in points
    team_in_points = points_positions['Team'].value_counts().idxmax()
    cars_in_points = points_positions['Team'].value_counts().max()
    
    # Add biggest mover statistics
    position_changes = predictions['Grid'] - predictions['Position']
    biggest_gain_idx = position_changes.idxmax()
    biggest_gain = predictions.loc[biggest_gain_idx]
    biggest_loss_idx = position_changes.idxmin()
    biggest_loss = predictions.loc[biggest_loss_idx]
    
    html += f"""
                    <div class="summary-card">
                        <h3>Notable Stats</h3>
                        <div><strong>Team with most cars in points:</strong> {team_in_points} ({cars_in_points})</div>
                        <div><strong>Biggest gain:</strong> {biggest_gain['Driver']} ({int(biggest_gain['Grid'])} → {int(biggest_gain['Position'])})</div>
                        <div><strong>Biggest loss:</strong> {biggest_loss['Driver']} ({int(biggest_loss['Grid'])} → {int(biggest_loss['Position'])})</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Predicted Results</h2>
                <table class="results-table">
                    <tr>
                        <th>Pos</th>
                        <th>Driver</th>
                        <th>Team</th>
                        <th>Grid</th>
                        <th>Interval</th>
                    </tr>
    """
    
    # Add all results
    for idx, row in predictions.iterrows():
        team_color = TEAM_COLORS.get(row['Team'], '#999999')
        html += f"""
                    <tr>
                        <td>{int(row['Position'])}</td>
                        <td><span class="team-color" style="background-color: {team_color};"></span> {row['Driver']}</td>
                        <td>{row['Team']}</td>
                        <td>{int(row['Grid'])}</td>
                        <td>{row['Interval']}</td>
                    </tr>
        """
    
    html += """
                </table>
            </div>
    """
    
    # Add visualization section
    html += """
            <div class="section">
                <h2>Race Visualizations</h2>
    """
    
    # Create dashboard visualization
    dashboard_path = create_interactive_dashboard(
        predictions, 
        track_name=track_name, 
        tire_data=tire_data
    )
    
    # Create relative path
    dashboard_rel_path = os.path.relpath(dashboard_path, os.path.dirname(save_path))
    
    html += f"""
                <div class="visualization">
                    <h3>Race Dashboard</h3>
                    <img src="{dashboard_rel_path}" alt="Race Dashboard">
                </div>
    """
    
    # Add Monte Carlo section if available
    if monte_carlo_data and 'probability_matrix' in monte_carlo_data:
        html += """
            <div class="section">
                <h2>Probability Analysis</h2>
                <p>Probability distribution of finishing positions based on Monte Carlo simulation:</p>
        """
        
        # Win probabilities
        if 'win_probabilities' in monte_carlo_data:
            html += """
                <h3>Win Probabilities</h3>
                <table class="results-table">
                    <tr>
                        <th>Driver</th>
                        <th>Win Probability</th>
                    </tr>
            """
            
            # Add top 5 drivers by win probability
            win_probs = monte_carlo_data['win_probabilities'].head(5)
            for driver, prob in win_probs.items():
                html += f"""
                    <tr>
                        <td>{driver}</td>
                        <td>{prob:.1%}</td>
                    </tr>
                """
            
            html += """
                </table>
            """
        
        # Podium probabilities
        if 'podium_probabilities' in monte_carlo_data:
            html += """
                <h3>Podium Probabilities</h3>
                <table class="results-table">
                    <tr>
                        <th>Driver</th>
                        <th>Podium Probability</th>
                    </tr>
            """
            
            # Add top 5 drivers by podium probability
            podium_probs = monte_carlo_data['podium_probabilities'].head(5)
            for driver, prob in podium_probs.items():
                html += f"""
                    <tr>
                        <td>{driver}</td>
                        <td>{prob:.1%}</td>
                    </tr>
                """
            
            html += """
                </table>
            """
        
        html += """
            </div>
        """
    
    # Add tire strategy section if available
    if tire_data:
        html += """
            <div class="section">
                <h2>Tire Strategy Analysis</h2>
        """
        
        # Add optimal stint information
        if 'optimal_stints' in tire_data:
            html += """
                <h3>Optimal Stint Lengths</h3>
                <table class="results-table">
                    <tr>
                        <th>Compound</th>
                        <th>Optimal Stint Length</th>
                    </tr>
            """
            
            for stint in tire_data['optimal_stints']:
                compound = stint['compound']
                display_name = stint['display_name']
                optimal_stint = stint['optimal_stint']
                
                html += f"""
                    <tr>
                        <td>{display_name} ({compound})</td>
                        <td>{optimal_stint} laps</td>
                    </tr>
                """
            
            html += """
                </table>
            """
        
        # Add crossover points
        if 'crossovers' in tire_data:
            html += """
                <h3>Tire Crossover Points</h3>
                <p>Points where one compound becomes faster than another due to degradation:</p>
                <table class="results-table">
                    <tr>
                        <th>Compounds</th>
                        <th>Crossover Lap</th>
                    </tr>
            """
            
            for key, data in tire_data['crossovers'].items():
                html += f"""
                    <tr>
                        <td>{data['name1']} ↔ {data['name2']}</td>
                        <td>Lap {data['lap']:.1f}</td>
                    </tr>
                """
            
            html += """
                </table>
            """
        
        # Add strategy recommendations
        html += """
            <h3>Strategy Recommendations</h3>
            <table class="results-table">
                <tr>
                    <th>Strategy Type</th>
                    <th>Recommended Strategy</th>
                </tr>
        """
        
        # Create sample strategy recommendations based on tire data
        if 'optimal_stints' in tire_data and len(tire_data['optimal_stints']) >= 2:
            # Extract compounds sorted by stint length (longest first)
            sorted_compounds = sorted(tire_data['optimal_stints'], key=lambda x: x['optimal_stint'], reverse=True)
            
            # Create standard strategy (longest stint first)
            standard = f"{sorted_compounds[0]['display_name']} → {sorted_compounds[1]['display_name']}"
            
            # Create aggressive strategy (shortest stint first)
            aggressive = f"{sorted_compounds[-1]['display_name']} → {sorted_compounds[-2]['display_name']} → {sorted_compounds[-1]['display_name']}"
            
            # Create conservative strategy (hardest compound for majority)
            hardest_idx = min(range(len(sorted_compounds)), key=lambda i: sorted_compounds[i]['display_name'] == 'SOFT')
            conservative = f"{sorted_compounds[hardest_idx]['display_name']} → {sorted_compounds[min(hardest_idx + 1, len(sorted_compounds) - 1)]['display_name']}"
            
            html += f"""
                <tr>
                    <td>Standard (1-stop)</td>
                    <td>{standard}</td>
                </tr>
                <tr>
                    <td>Aggressive (2-stop)</td>
                    <td>{aggressive}</td>
                </tr>
                <tr>
                    <td>Conservative (1-stop)</td>
                    <td>{conservative}</td>
                </tr>
            """
        else:
            html += """
                <tr>
                    <td colspan="2">Insufficient tire data for strategy recommendations</td>
                </tr>
            """
        
        html += """
                </table>
            </div>
        """
    
    # Add historical accuracy section if available
    if historical_data and 'accuracy_metrics' in historical_data:
        html += """
            <div class="section">
                <h2>Model Accuracy</h2>
                <p>Historical accuracy of the prediction model:</p>
                <table class="results-table">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
        """
        
        metrics = historical_data['accuracy_metrics']
        
        if 'mean_absolute_error' in metrics:
            html += f"""
                <tr>
                    <td>Mean Absolute Error</td>
                    <td>{metrics['mean_absolute_error']:.2f} positions</td>
                </tr>
            """
        
        if 'correct_percentage' in metrics:
            html += f"""
                <tr>
                    <td>Exact Position Accuracy</td>
                    <td>{metrics['correct_percentage']:.1f}%</td>
                </tr>
            """
        
        if 'within_one_percentage' in metrics:
            html += f"""
                <tr>
                    <td>Within ±1 Position</td>
                    <td>{metrics['within_one_percentage']:.1f}%</td>
                </tr>
            """
        
        html += """
                </table>
            </div>
        """
    
    # Add footer
    html += """
            <div class="footnote">
                <p>This report was generated using the F1 Race Prediction Model. All predictions are based on statistical analysis and simulation.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(save_path, 'w') as f:
        f.write(html)
    
    print(f"Race report saved to {save_path}")
    
    return save_path

# In visualization/dashboard.py - Simplify the dashboard creation
def create_simple_dashboard(predictions, track_name, save_path=None):
    """Create a simplified dashboard with minimal dependencies."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # Create figure with fewer subplots
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot 1: Top 10 Finishers (simplified)
    ax1 = fig.add_subplot(gs[0, 0])
    top_10 = predictions.head(10).copy()
    bars = ax1.barh(top_10['Driver'], top_10['Interval (s)'])
    ax1.set_title('Predicted Top 10 Finishers')
    
    # Plot 2: Grid vs. Finish (simplified)
    ax2 = fig.add_subplot(gs[0, 1])
    for i, row in predictions.iterrows():
        ax2.scatter(row['Grid'], row['Position'])
        ax2.text(row['Grid'] + 0.1, row['Position'] + 0.1, row['Driver'], fontsize=8)
    ax2.set_title('Grid vs. Predicted Finish')
    
    # Save with proper error handling
    if save_path is None:
        save_path = f"dashboard_{track_name.lower().replace(' ', '_')}.png"
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    
    return save_path