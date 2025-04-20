"""
Constants for F1 race predictions.

This module contains constant definitions used across the prediction system,
including team colors, tire compounds, track characteristics, and performance factors.
"""
"""
Helper functions and constants for F1 race predictions.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

DRIVERS = {
            'Max Verstappen': 'Red Bull Racing', 
            'Yuki Tsunoda': 'Red Bull Racing', 
            'Charles Leclerc': 'Ferrari', 
            'Lewis Hamilton': 'Ferrari',
            'George Russell': 'Mercedes', 
            'Kimi Antonelli': 'Mercedes', 
            'Lando Norris': 'McLaren', 
            'Oscar Piastri': 'McLaren',
            'Fernando Alonso': 'Aston Martin', 
            'Lance Stroll': 'Aston Martin', 
            'Liam Lawson': 'Racing Bulls', 
            'Isack Hadjar': 'Racing Bulls',
            'Alexander Albon': 'Williams', 
            'Carlos Sainz': 'Williams', 
            'Gabriel Bortoleto': 'Kick Sauber', 
            'Nico Hulkenberg': 'Kick Sauber',
            'Oliver Bearman': 'Haas F1 Team', 
            'Esteban Ocon': 'Haas F1 Team', 
            'Jack Doohan': 'Alpine', 
            'Pierre Gasly': 'Alpine'
        }
            
TEAMS = [
            'Red Bull Racing', 'Ferrari',
            'Mercedes', 'McLaren',
            'Aston Martin','Racing Bulls',
            'Williams','Kick Sauber',
            'Haas F1 Team', 'Alpine'
        ]
# Team colors for visualization
TEAM_COLORS = {
    'Red Bull Racing': '#0600EF',
    'Ferrari': '#DC0000',
    'Mercedes': '#00D2BE',
    'McLaren': '#FF8700',
    'Aston Martin': '#006F62',
    'Alpine': '#0090FF',
    'Williams': '#005AFF',
    'RB': '#1E41FF',
    'Racing Bulls': '#1E41FF',  # Same as RB
    'Haas F1 Team': '#FFFFFF',
    'Kick Sauber': '#900000'
}

# Performance factors for 2025 season
PERFORMANCE_FACTORS = {
    'team_factors': {
        'Red Bull Racing': 0.98,  # Strong but not as dominant
        'Ferrari': 0.99,          # Strengthened position
        'McLaren': 0.98,          # Very strong - championship contender
        'Mercedes': 0.99,         # Improving through the season
        'Aston Martin': 1.01,     # Midfield
        'Racing Bulls': 1.01,     # Better than RB, accounting for rookies
        'RB': 1.02,               # Upper midfield
        'Williams': 1.03,         # Struggling
        'Haas F1 Team': 1.02,     # Improved from previous seasons
        'Kick Sauber': 1.03,      # Back of the grid but with rookie talent
        'Alpine': 1.03,           # Struggling with rookie potential
    },
    'driver_factors': {
        # Established drivers
        'Max Verstappen': 0.97,   # Exceptional driver, championship leader
        'Yuki Tsunoda': 1.01,     # Inconsistent
        'Charles Leclerc': 0.98,  # Very strong
        'Carlos Sainz': 0.99,     # Strong and consistent
        'Lewis Hamilton': 0.99,   # Still one of the best
        'George Russell': 0.99,   # Fast and improving
        'Lando Norris': 0.98,     # Championship contender
        'Oscar Piastri': 0.99,    # Fast improving sophomore
        'Fernando Alonso': 1.00,  # Experienced but car limited
        'Lance Stroll': 1.02,     # Inconsistent
        'Alexander Albon': 1.01,  # Outperforming the car
        'Logan Sargeant': 1.04,   # Struggling
        'Valtteri Bottas': 1.02,  # Experienced but car limited
        'Zhou Guanyu': 1.03,      # Inconsistent
        'Kevin Magnussen': 1.02,  # Experienced midfield driver
        'Nico Hulkenberg': 1.01,  # Strong qualifying, weaker races
        'Pierre Gasly': 1.02,     # Car limited
        'Esteban Ocon': 1.02,     # Car limited
        
        # Rookies with fair chance factors
        'Kimi Antonelli': 1.00,   # Mercedes rookie with high potential
        'Oliver Bearman': 1.01,   # Haas rookie who impressed in Ferrari substitute role
        'Gabriel Bortoleto': 1.01, # Kick Sauber rookie with promise
        'Jack Doohan': 1.01,      # Alpine rookie with potential
        'Isack Hadjar': 1.01,     # Racing Bulls rookie talent
        'Liam Lawson': 1.00,      # Racing Bulls rookie who already impressed in earlier outings
    }
}

# List of rookie drivers
ROOKIES = [
    'Kimi Antonelli', 'Oliver Bearman', 'Gabriel Bortoleto', 
    'Jack Doohan', 'Isack Hadjar', 'Liam Lawson'
]

# Constants for tire compounds
PIRELLI_COMPOUNDS = {
    'C1': {
        'color': '#FFFFFF',  # White
        'initial_advantage': 0.3,  # Slowest initially (seconds)
        'deg_rate': 0.005,   # Lowest degradation (seconds per lap)
        'thermal_sensitivity': 0.5,  # Lower value = less sensitive to temperature
        'wear_rate': 0.07,   # Lower value = less physical wear
        'grip_level': 0.7,   # Lower value = less grip
        'optimal_temp_range': (110, 140),  # Higher operating temperature (°C)
        'optimal_tracks': ['Silverstone', 'Barcelona', 'Paul Ricard'],  # Tracks where this compound excels
        'description': 'Hardest compound, very low degradation, used at high-load/temperature tracks'
    },
    'C2': {
        'color': '#FFFF00',  # Yellow
        'initial_advantage': 0.1,
        'deg_rate': 0.008,
        'thermal_sensitivity': 0.6,
        'wear_rate': 0.10,
        'grip_level': 0.8,
        'optimal_temp_range': (105, 135),
        'optimal_tracks': ['Bahrain', 'Jeddah', 'Shanghai'],
        'description': 'Hard compound, low degradation, good for abrasive tracks'
    },
    'C3': {
        'color': '#FFFFFF',  # White (with white markings)
        'initial_advantage': -0.1,
        'deg_rate': 0.015,
        'thermal_sensitivity': 0.7,
        'wear_rate': 0.15,
        'grip_level': 0.85,
        'optimal_temp_range': (95, 125),
        'optimal_tracks': ['Melbourne', 'Montreal', 'Monza'],
        'description': 'Medium compound, balanced degradation, versatile across circuits'
    },
    'C4': {
        'color': '#FFFF00',  # Yellow (with yellow markings)
        'initial_advantage': -0.5,
        'deg_rate': 0.025,
        'thermal_sensitivity': 0.8,
        'wear_rate': 0.20,
        'grip_level': 0.9,
        'optimal_temp_range': (85, 115),
        'optimal_tracks': ['Monaco', 'Baku', 'Singapore'],
        'description': 'Soft compound, moderate degradation, good for cooler conditions'
    },
    'C5': {
        'color': '#FF0000',  # Red
        'initial_advantage': -0.9,
        'deg_rate': 0.04,
        'thermal_sensitivity': 0.9,
        'wear_rate': 0.25,
        'grip_level': 1.0,
        'optimal_temp_range': (80, 110),
        'optimal_tracks': ['Monaco', 'Singapore', 'Las Vegas'],
        'description': 'Softest compound, highest degradation, best for qualifying/short stints'
    }
}

# Track surface characteristics affecting tire wear
TRACK_CHARACTERISTICS = {
    'Bahrain': {
        'abrasiveness': 0.8,  # How abrasive the surface is (0-1)
        'temperature': 35,    # Average track temperature (°C)
        'layout_stress': 0.7, # How much the layout stresses tires (0-1)
        'selected_compounds': ['C1', 'C2', 'C3'],  # Compounds available for the race weekend
        'description': 'High temperatures and abrasive surface lead to high tire degradation'
    },
    'Jeddah': {
        'abrasiveness': 0.5,
        'temperature': 30,
        'layout_stress': 0.8,
        'selected_compounds': ['C2', 'C3', 'C4'],
        'description': 'Smooth surface but high speeds create thermal degradation'
    },
    'Melbourne': {
        'abrasiveness': 0.4,
        'temperature': 25,
        'layout_stress': 0.6,
        'selected_compounds': ['C2', 'C3', 'C4'],
        'description': 'Relatively smooth surface with moderate temperatures'
    },
    'Imola': {
        'abrasiveness': 0.7,
        'temperature': 22,
        'layout_stress': 0.6,
        'selected_compounds': ['C2', 'C3', 'C4'],
        'description': 'Traditional circuit with moderate grip levels'
    },
    'Miami': {
        'abrasiveness': 0.5,
        'temperature': 32,
        'layout_stress': 0.7,
        'selected_compounds': ['C2', 'C3', 'C4'],
        'description': 'New surface with high temperatures'
    },
    'Monaco': {
        'abrasiveness': 0.3,
        'temperature': 22,
        'layout_stress': 0.4,
        'selected_compounds': ['C3', 'C4', 'C5'],
        'description': 'Smooth surface with low degradation due to low speeds'
    },
    'Barcelona': {
        'abrasiveness': 0.7,
        'temperature': 28,
        'layout_stress': 0.8,
        'selected_compounds': ['C1', 'C2', 'C3'],
        'description': 'High-energy corners create significant tire stress'
    },
    'Montreal': {
        'abrasiveness': 0.5,
        'temperature': 20,
        'layout_stress': 0.6,
        'selected_compounds': ['C3', 'C4', 'C5'],
        'description': 'Variable temperatures and stop-start nature stress tires'
    },
    'Silverstone': {
        'abrasiveness': 0.8,
        'temperature': 24,
        'layout_stress': 0.9,
        'selected_compounds': ['C1', 'C2', 'C3'],
        'description': 'High-speed corners create extreme loads on tires'
    },
    'Hungaroring': {
        'abrasiveness': 0.6,
        'temperature': 30,
        'layout_stress': 0.5,
        'selected_compounds': ['C2', 'C3', 'C4'],
        'description': 'Twisty layout with high temperatures'
    },
    'Spa': {
        'abrasiveness': 0.7,
        'temperature': 22,
        'layout_stress': 0.8,
        'selected_compounds': ['C1', 'C2', 'C3'],
        'description': 'High loads and variable weather conditions'
    },
    'Zandvoort': {
        'abrasiveness': 0.6,
        'temperature': 23,
        'layout_stress': 0.7,
        'selected_compounds': ['C1', 'C2', 'C3'],
        'description': 'Banking creates unique tire stresses'
    },
    'Monza': {
        'abrasiveness': 0.5,
        'temperature': 26,
        'layout_stress': 0.7,
        'selected_compounds': ['C2', 'C3', 'C4'],
        'description': 'High speeds and low downforce create thermal challenges'
    },
    'Singapore': {
        'abrasiveness': 0.6,
        'temperature': 32,
        'layout_stress': 0.6,
        'selected_compounds': ['C3', 'C4', 'C5'],
        'description': 'High humidity and temperatures with a bumpy surface'
    },
    'Suzuka': {
        'abrasiveness': 0.7,
        'temperature': 24,
        'layout_stress': 0.9,
        'selected_compounds': ['C1', 'C2', 'C3'],
        'description': 'High-speed layout with 8-figure creating lateral forces'
    },
    'Austin': {
        'abrasiveness': 0.8,
        'temperature': 26,
        'layout_stress': 0.8,
        'selected_compounds': ['C2', 'C3', 'C4'],
        'description': 'Bumpy surface with high-energy corners'
    },
    'Mexico City': {
        'abrasiveness': 0.5,
        'temperature': 25,
        'layout_stress': 0.6,
        'selected_compounds': ['C2', 'C3', 'C4'],
        'description': 'High altitude affects downforce and tire behavior'
    },
    'Sao Paulo': {
        'abrasiveness': 0.6,
        'temperature': 28,
        'layout_stress': 0.7,
        'selected_compounds': ['C2', 'C3', 'C4'],
        'description': 'Undulating layout with variable weather'
    },
    'Las Vegas': {
        'abrasiveness': 0.4,
        'temperature': 15,
        'layout_stress': 0.6,
        'selected_compounds': ['C3', 'C4', 'C5'],
        'description': 'Cold night race with long straights and sharp corners'
    },
    'Qatar': {
        'abrasiveness': 0.7,
        'temperature': 33,
        'layout_stress': 0.8,
        'selected_compounds': ['C1', 'C2', 'C3'],
        'description': 'High speeds and temperatures create extreme tire stress'
    },
    'Abu Dhabi': {
        'abrasiveness': 0.6,
        'temperature': 28,
        'layout_stress': 0.7,
        'selected_compounds': ['C2', 'C3', 'C4'],
        'description': 'Smooth surface with cooling track temperatures'
    }
}

# Team colors for visualization
TEAM_COLORS = {
    'Red Bull Racing': '#0600EF',
    'Ferrari': '#DC0000',
    'Mercedes': '#00D2BE',
    'McLaren': '#FF8700',
    'Aston Martin': '#006F62',
    'Alpine': '#0090FF',
    'Williams': '#005AFF',
    'Racing Bulls': '#1E41FF',
    'Haas F1 Team': '#FFFFFF',
    'Kick Sauber': '#900000'
}

# Rookie drivers for 2025
ROOKIES = [
    'Kimi Antonelli', 'Oliver Bearman', 'Gabriel Bortoleto', 
    'Jack Doohan', 'Isack Hadjar', 'Liam Lawson'
]

# Performance factors for teams and drivers (lower = better)
TEAM_PERFORMANCE = {
    'Red Bull Racing': 0.98,
    'Ferrari': 0.99,
    'McLaren': 0.98,
    'Mercedes': 0.99,
    'Aston Martin': 1.01,
    'Racing Bulls': 1.01,
    'Williams': 1.03,
    'Haas F1 Team': 1.02,
    'Kick Sauber': 1.03,
    'Alpine': 1.03
}

DRIVER_PERFORMANCE = {
    'Max Verstappen': 0.97,
    'Charles Leclerc': 0.98,
    'Lando Norris': 0.98,
    'Lewis Hamilton': 0.99,
    'Carlos Sainz': 0.99,
    'George Russell': 0.99,
    'Oscar Piastri': 0.99,
    'Fernando Alonso': 1.00,
    'Kimi Antonelli': 1.00,
    'Liam Lawson': 1.00,
    'Alexander Albon': 1.01,
    'Yuki Tsunoda': 1.01,
    'Nico Hulkenberg': 1.01,
    'Oliver Bearman': 1.01,
    'Gabriel Bortoleto': 1.01,
    'Jack Doohan': 1.01,
    'Isack Hadjar': 1.01,
    'Lance Stroll': 1.02,
    'Esteban Ocon': 1.02,
    'Pierre Gasly': 1.02
}

def ensure_directory(directory_path):
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path (str): Path to directory
        
    Returns:
        str: Path to directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
    return directory_path

def get_team_color(team, default='gray'):
    """
    Get color for a team.
    
    Args:
        team (str): Team name
        default (str): Default color
        
    Returns:
        str: Hex color code
    """
    return TEAM_COLORS.get(team, default)

def get_team_for_driver(driver):
    """
    Get the team for a driver.
    
    Args:
        driver (str): Driver name
        
    Returns:
        str: Team name or None if driver not found
    """
    return DRIVERS.get(driver)

def save_figure(fig, filename, directory='results'):
    """
    Save a matplotlib figure to disk.
    
    Args:
        fig (Figure): Matplotlib figure
        filename (str): Filename
        directory (str): Directory to save to
        
    Returns:
        str: Path to saved figure
    """
    # Ensure directory exists
    ensure_directory(directory)
    
    # Add timestamp to filename to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_with_timestamp = f"{timestamp}_{filename}"
    
    # Create full path
    filepath = os.path.join(directory, filename_with_timestamp)
    
    # Save figure
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Figure saved to {filepath}")
    
    return filepath