"""
Constants for F1 race predictions.

This module contains constant definitions used across the prediction system,
including team colors, tire compounds, track characteristics, and performance factors.
It also provides functionality to fetch the official F1 event schedule.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Set up logging
logger = logging.getLogger('f1_prediction.constants')

# Try to import fastf1, handle if not available
try:
    import fastf1
    from fastf1.events import Event, EventSchedule
    FASTF1_AVAILABLE = True
except ImportError:
    FASTF1_AVAILABLE = False
    logger.warning("FastF1 not available. Using default schedule.")

# Current drivers and teams for 2025 season
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
    'Red Bull Racing', 'Ferrari', 'Mercedes', 'McLaren',
    'Aston Martin', 'Racing Bulls', 'Williams', 'Kick Sauber',
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
        'Valtteri Bottas': 1.02,  # Experienced but car limited
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

# Default event schedule - used as fallback if FastF1 unavailable
DEFAULT_2025_SCHEDULE = [
    {"round": 1, "name": "Bahrain Grand Prix", "circuit": "Bahrain International Circuit", "date": "2025-03-08"},
    {"round": 2, "name": "Saudi Arabian Grand Prix", "circuit": "Jeddah Corniche Circuit", "date": "2025-03-15"},
    {"round": 3, "name": "Australian Grand Prix", "circuit": "Albert Park Circuit", "date": "2025-03-29"},
    {"round": 4, "name": "Japanese Grand Prix", "circuit": "Suzuka International Racing Course", "date": "2025-04-12"},
    {"round": 5, "name": "Chinese Grand Prix", "circuit": "Shanghai International Circuit", "date": "2025-04-19"},
    {"round": 6, "name": "Miami Grand Prix", "circuit": "Miami International Autodrome", "date": "2025-05-03"},
    {"round": 7, "name": "Emilia Romagna Grand Prix", "circuit": "Autodromo Enzo e Dino Ferrari", "date": "2025-05-17"},
    {"round": 8, "name": "Monaco Grand Prix", "circuit": "Circuit de Monaco", "date": "2025-05-31"},
    {"round": 9, "name": "Canadian Grand Prix", "circuit": "Circuit Gilles Villeneuve", "date": "2025-06-14"},
    {"round": 10, "name": "Spanish Grand Prix", "circuit": "Circuit de Barcelona-Catalunya", "date": "2025-06-28"},
    {"round": 11, "name": "Austrian Grand Prix", "circuit": "Red Bull Ring", "date": "2025-07-05"},
    {"round": 12, "name": "British Grand Prix", "circuit": "Silverstone Circuit", "date": "2025-07-19"},
    {"round": 13, "name": "Hungarian Grand Prix", "circuit": "Hungaroring", "date": "2025-08-02"},
    {"round": 14, "name": "Belgian Grand Prix", "circuit": "Circuit de Spa-Francorchamps", "date": "2025-08-30"},
    {"round": 15, "name": "Dutch Grand Prix", "circuit": "Circuit Zandvoort", "date": "2025-09-06"},
    {"round": 16, "name": "Italian Grand Prix", "circuit": "Autodromo Nazionale Monza", "date": "2025-09-13"},
    {"round": 17, "name": "Azerbaijan Grand Prix", "circuit": "Baku City Circuit", "date": "2025-09-27"},
    {"round": 18, "name": "Singapore Grand Prix", "circuit": "Marina Bay Street Circuit", "date": "2025-10-04"},
    {"round": 19, "name": "United States Grand Prix", "circuit": "Circuit of the Americas", "date": "2025-10-18"},
    {"round": 20, "name": "Mexican Grand Prix", "circuit": "Autódromo Hermanos Rodríguez", "date": "2025-10-25"},
    {"round": 21, "name": "Brazilian Grand Prix", "circuit": "Autódromo José Carlos Pace", "date": "2025-11-08"},
    {"round": 22, "name": "Las Vegas Grand Prix", "circuit": "Las Vegas Strip Circuit", "date": "2025-11-22"},
    {"round": 23, "name": "Qatar Grand Prix", "circuit": "Lusail International Circuit", "date": "2025-11-29"},
    {"round": 24, "name": "Abu Dhabi Grand Prix", "circuit": "Yas Marina Circuit", "date": "2025-12-06"}
]

class F1Schedule:
    """
    Class to handle F1 race schedule data using FastF1 API.
    Also provides fallback if FastF1 is unavailable.
    """
    
    def __init__(self):
        """Initialize the F1 schedule handler."""
        self.schedule_2025 = None
        self.event_formats = {}
        self.session_info = {}
        
        # Try to load schedule from FastF1
        self.load_schedule()
    
    def load_schedule(self, year=2025):
        """
        Load F1 schedule for specified year using FastF1.
        Falls back to default schedule if FastF1 unavailable.
        
        Args:
            year (int): Season year
            
        Returns:
            bool: Success status
        """
        if FASTF1_AVAILABLE:
            try:
                # Load schedule using FastF1
                self.schedule_2025 = fastf1.get_event_schedule(year)
                logger.info(f"Loaded {year} F1 schedule from FastF1")
                
                # Extract event formats
                for _, event in self.schedule_2025.iterrows():
                    round_num = event['RoundNumber']
                    self.event_formats[round_num] = event['EventFormat']
                    
                    # Get session dates and types
                    sessions = {}
                    for i in range(1, 6):  # Up to 5 sessions
                        session_key = f'Session{i}'
                        date_key = f'Session{i}Date'
                        date_key_utc = f'Session{i}DateUtc'
                        
                        if session_key in event and not pd.isna(event[session_key]):
                            sessions[event[session_key]] = {
                                'date': event.get(date_key, None),
                                'date_utc': event.get(date_key_utc, None),
                            }
                    
                    self.session_info[round_num] = sessions
                
                return True
            except Exception as e:
                logger.warning(f"Failed to load schedule from FastF1: {e}")
                logger.warning("Falling back to default schedule")
                self._load_default_schedule(year)
                return False
        else:
            logger.warning("FastF1 not available, using default schedule")
            self._load_default_schedule(year)
            return False
    
    def _load_default_schedule(self, year=2025):
        """
        Load default hardcoded schedule as fallback.
        
        Args:
            year (int): Season year
        """
        # Create a DataFrame from the default schedule
        schedule_data = DEFAULT_2025_SCHEDULE
        
        # Convert to DataFrame
        self.schedule_2025 = pd.DataFrame(schedule_data)
        
        # Set default event formats (conventional for all)
        for round_num in self.schedule_2025['round'].values:
            self.event_formats[round_num] = 'conventional'
            
            # Create default session structure
            race_date = pd.to_datetime(
                self.schedule_2025[self.schedule_2025['round'] == round_num]['date'].iloc[0]
            )
            
            # Assuming conventional format: FP1/FP2 on Friday, FP3/Quali on Saturday, Race on Sunday
            self.session_info[round_num] = {
                'Practice 1': {
                    'date': race_date - pd.Timedelta(days=2),
                    'date_utc': race_date - pd.Timedelta(days=2)
                },
                'Practice 2': {
                    'date': race_date - pd.Timedelta(days=2),
                    'date_utc': race_date - pd.Timedelta(days=2)
                },
                'Practice 3': {
                    'date': race_date - pd.Timedelta(days=1),
                    'date_utc': race_date - pd.Timedelta(days=1)
                },
                'Qualifying': {
                    'date': race_date - pd.Timedelta(days=1),
                    'date_utc': race_date - pd.Timedelta(days=1)
                },
                'Race': {
                    'date': race_date,
                    'date_utc': race_date
                }
            }
    
    def get_event_schedule(self, year=2025):
        """
        Get the event schedule for a specific year.
        
        Args:
            year (int): Season year
            
        Returns:
            DataFrame: Event schedule
        """
        if self.schedule_2025 is None or year != 2025:
            self.load_schedule(year)
        
        return self.schedule_2025
    
    def get_event(self, round_num, year=2025):
        """
        Get event information for a specific round.
        
        Args:
            round_num (int): Round number
            year (int): Season year
            
        Returns:
            dict: Event information
        """
        if self.schedule_2025 is None or year != 2025:
            self.load_schedule(year)
        
        if FASTF1_AVAILABLE:
            try:
                # Get event using FastF1
                event = fastf1.get_event(year, round_num)
                
                # Convert to dictionary for consistency
                event_info = {
                    'name': event['EventName'],
                    'round': event['RoundNumber'],
                    'date': event['EventDate'],
                    'format': event['EventFormat'],
                    'circuit': event['Location'],
                    'country': event['Country'],
                    'sessions': {}
                }
                
                # Extract session information
                for i in range(1, 6):  # Up to 5 sessions
                    session_key = f'Session{i}'
                    date_key = f'Session{i}Date'
                    date_key_utc = f'Session{i}DateUtc'
                    
                    if session_key in event and not pd.isna(event[session_key]):
                        event_info['sessions'][event[session_key]] = {
                            'date': event.get(date_key, None),
                            'date_utc': event.get(date_key_utc, None),
                        }
                
                return event_info
            except Exception as e:
                logger.warning(f"Failed to get event from FastF1: {e}")
                logger.warning("Falling back to default event info")
        
        # Fallback to our own data
        event_row = self.schedule_2025[self.schedule_2025['round'] == round_num]
        
        if len(event_row) == 0:
            raise ValueError(f"No event found for round {round_num}")
        
        event_info = {
            'name': event_row['name'].iloc[0],
            'round': round_num,
            'date': pd.to_datetime(event_row['date'].iloc[0]),
            'format': self.event_formats.get(round_num, 'conventional'),
            'circuit': event_row['circuit'].iloc[0],
            'country': event_row['circuit'].iloc[0].split(' ')[0],  # Approximate
            'sessions': self.session_info.get(round_num, {})
        }
        
        return event_info
    
    def get_session_date(self, round_num, session_identifier, year=2025):
        """
        Get the date for a specific session.
        
        Args:
            round_num (int): Round number
            session_identifier (str): Session identifier (e.g., 'FP1', 'Q', 'R')
            year (int): Season year
            
        Returns:
            datetime: Session date
        """
        if FASTF1_AVAILABLE:
            try:
                event = fastf1.get_event(year, round_num)
                
                # Map short identifiers to full names
                session_map = {
                    'FP1': 'Practice 1',
                    'FP2': 'Practice 2', 
                    'FP3': 'Practice 3',
                    'Q': 'Qualifying',
                    'SQ': 'Sprint Qualifying',
                    'SS': 'Sprint Shootout',
                    'S': 'Sprint',
                    'R': 'Race'
                }
                
                # Convert identifier if needed
                if session_identifier in session_map:
                    session_identifier = session_map[session_identifier]
                
                # Get session date
                return event.get_session_date(session_identifier)
            except Exception as e:
                logger.warning(f"Failed to get session date from FastF1: {e}")
                logger.warning("Falling back to default session date calculation")
        
        # Fallback to our own data
        event_info = self.get_event(round_num, year)
        
        # Map short identifiers to full names
        session_map = {
            'FP1': 'Practice 1',
            'FP2': 'Practice 2', 
            'FP3': 'Practice 3',
            'Q': 'Qualifying',
            'SQ': 'Sprint Qualifying',
            'SS': 'Sprint Shootout',
            'S': 'Sprint',
            'R': 'Race'
        }
        
        # Convert identifier if needed
        if session_identifier in session_map:
            session_identifier = session_map[session_identifier]
        
        # Get session date from our data
        if session_identifier in event_info['sessions']:
            return event_info['sessions'][session_identifier]['date']
        
        # If not found, calculate based on race day
        race_date = event_info['date']
        
        if session_identifier == 'Race':
            return race_date
        elif session_identifier in ['Qualifying', 'Practice 3']:
            return race_date - pd.Timedelta(days=1)
        elif session_identifier in ['Practice 1', 'Practice 2']:
            return race_date - pd.Timedelta(days=2)
        elif session_identifier in ['Sprint', 'Sprint Shootout', 'Sprint Qualifying']:
            return race_date - pd.Timedelta(days=1)
        else:
            raise ValueError(f"Unknown session identifier: {session_identifier}")

    def check_sprint_race(self, round_num, year=2025):
        """
        Check if a race weekend includes a sprint race.
        
        Args:
            round_num (int): Round number
            year (int): Season year
            
        Returns:
            bool: True if the event includes a sprint race
        """
        event_format = self.event_formats.get(round_num, 'conventional')
        return 'sprint' in event_format

    def get_race_info(self, round_num, year=2025):
        """
        Get comprehensive race information suitable for prediction.
        
        Args:
            round_num (int): Round number
            year (int): Season year
            
        Returns:
            dict: Race information with track, format, and prediction-relevant data
        """
        event_info = self.get_event(round_num, year)
        
        # Find track in TRACK_CHARACTERISTICS or use approximate match
        track_name = event_info['circuit']
        track_key = self._find_matching_track(track_name)
        
        # Get track characteristics
        track_chars = TRACK_CHARACTERISTICS.get(track_key, {})
        
        # Default safety car and rain probabilities based on track characteristics
        safety_car_prob = track_chars.get('layout_stress', 0.6)
        rain_prob = 0.0  # Default
        
        # Calculate rain probability if we have temperature data
        if 'temperature' in track_chars:
            # Lower temperatures are associated with higher rain probability
            temp = track_chars['temperature']
            rain_prob = max(0, min(0.6, (30 - temp) / 40))
        
        # Compile race information
        race_info = {
            "track": track_name,
            "name": event_info['name'],
            "country": event_info['country'],
            "date": event_info['sessions'].get('Race', {}).get('date', event_info['date']),
            "format": event_info['format'],
            "has_sprint": self.check_sprint_race(round_num),
            "safety_car_prob": safety_car_prob,
            "rain_prob": rain_prob,
            "track_characteristics": track_chars,
            "sessions": {}
        }
        
        # Add session data
        for session_name, session_data in event_info['sessions'].items():
            race_info['sessions'][session_name] = {
                'date': session_data.get('date', None),
                'date_utc': session_data.get('date_utc', None)
            }
        
        return race_info

    def _find_matching_track(self, track_name):
        """
        Find the closest matching track in TRACK_CHARACTERISTICS.
        
        Args:
            track_name (str): Track name to match
            
        Returns:
            str: Matching track key from TRACK_CHARACTERISTICS or original if no match
        """
        # Direct match
        if track_name in TRACK_CHARACTERISTICS:
            return track_name
        
        # Try fuzzy matching
        for known_track in TRACK_CHARACTERISTICS.keys():
            if (track_name.lower() in known_track.lower() or 
                known_track.lower() in track_name.lower()):
                return known_track
        
        # Try matching with common mappings
        track_aliases = {
            "Melbourne": "Melbourne",
            "Albert Park": "Melbourne",
            "Bahrain International Circuit": "Bahrain",
            "Jeddah Corniche Circuit": "Jeddah",
            "Shanghai International Circuit": "Shanghai",
            "Miami International Autodrome": "Miami",
            "Imola": "Imola",
            "Autodromo Enzo e Dino Ferrari": "Imola",
            "Monaco": "Monaco",
            "Circuit de Monaco": "Monaco",
            "Circuit Gilles Villeneuve": "Montreal",
            "Barcelona": "Barcelona",
            "Circuit de Barcelona-Catalunya": "Barcelona",
            "Red Bull Ring": "Spielberg",
            "Silverstone": "Silverstone",
            "Silverstone Circuit": "Silverstone",
            "Hungaroring": "Hungaroring",
            "Spa": "Spa",
            "Circuit de Spa-Francorchamps": "Spa",
            "Zandvoort": "Zandvoort",
            "Circuit Zandvoort": "Zandvoort",
            "Monza": "Monza",
            "Autodromo Nazionale Monza": "Monza",
            "Baku City Circuit": "Baku",
            "Marina Bay Street Circuit": "Singapore",
            "Circuit of the Americas": "Austin",
            "COTA": "Austin",
            "Autódromo Hermanos Rodríguez": "Mexico City",
            "Autódromo José Carlos Pace": "Sao Paulo",
            "Interlagos": "Sao Paulo",
            "Las Vegas Strip Circuit": "Las Vegas",
            "Lusail International Circuit": "Qatar",
            "Yas Marina Circuit": "Abu Dhabi"
        }
        
        if track_name in track_aliases:
            alias = track_aliases[track_name]
            if alias in TRACK_CHARACTERISTICS:
                return alias
        
        # No match found, return original
        return track_name

    def get_upcoming_races(self, current_date=None, count=5):
        """
        Get the next upcoming races from the schedule.
        
        Args:
            current_date (datetime, optional): Reference date (defaults to today)
            count (int, optional): Number of upcoming races to return
            
        Returns:
            DataFrame: Upcoming races information
        """
        if current_date is None:
            current_date = pd.Timestamp.now()
        
        if self.schedule_2025 is None:
            self.load_schedule()
        
        # Create a copy with parsed dates
        schedule_copy = self.schedule_2025.copy()
        
        # Convert string dates to datetime if needed
        if 'date' in schedule_copy.columns and isinstance(schedule_copy['date'].iloc[0], str):
            schedule_copy['date'] = pd.to_datetime(schedule_copy['date'])
        
        # Sort by date
        schedule_copy = schedule_copy.sort_values('date')
        
        # Filter to upcoming races
        if FASTF1_AVAILABLE:
            # Use EventDate for comparison
            upcoming = schedule_copy[schedule_copy['EventDate'] >= current_date]
        else:
            # Use our date column
            upcoming = schedule_copy[schedule_copy['date'] >= current_date]
        
        # Return requested number of races
        return upcoming.head(count)

    # Add these methods to the F1Schedule class
    F1Schedule.check_sprint_race = check_sprint_race
    F1Schedule.get_race_info = get_race_info
    F1Schedule._find_matching_track = _find_matching_track
    F1Schedule.get_upcoming_races = get_upcoming_races

    # Create a global instance
    f1_schedule = F1Schedule()

    # Helper functions to use with the global instance
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
            logger.info(f"Created directory: {directory_path}")
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
        logger.info(f"Figure saved to {filepath}")
        
        return filepath

    def get_race_info(round_num, year=2025):
        """
        Get race information for a specific round.
        Shorthand for using the global f1_schedule instance.
        
        Args:
            round_num (int): Round number
            year (int): Season year
            
        Returns:
            dict: Race information
        """
        return f1_schedule.get_race_info(round_num, year)

    def get_upcoming_races(count=5):
        """
        Get upcoming races from the schedule.
        Shorthand for using the global f1_schedule instance.
        
        Args:
            count (int): Number of upcoming races to return
            
        Returns:
            DataFrame: Upcoming races information
        """
        return f1_schedule.get_upcoming_races(count=count)

    def find_race_by_name(race_name, year=2025):
        """
        Find a race by its name.
        
        Args:
            race_name (str): Race name to search for
            year (int): Season year
            
        Returns:
            dict: Race information or None if not found
        """
        if FASTF1_AVAILABLE:
            try:
                # Use FastF1's get_event_by_name
                event = fastf1.get_event_by_name(race_name, year)
                round_num = event['RoundNumber']
                return get_race_info(round_num, year)
            except Exception as e:
                logger.warning(f"Could not find race by name using FastF1: {e}")
        
        # Fallback to our own search
        schedule = f1_schedule.get_event_schedule(year)
        
        # Try direct match first
        matches = schedule[schedule['name'].str.contains(race_name, case=False)]
        
        if len(matches) > 0:
            round_num = matches.iloc[0]['round']
            return get_race_info(round_num, year)
        
        # Try fuzzy match with circuit name
        matches = schedule[schedule['circuit'].str.contains(race_name, case=False)]
        
        if len(matches) > 0:
            round_num = matches.iloc[0]['round']
            return get_race_info(round_num, year)
        
        # No matches found
        return None