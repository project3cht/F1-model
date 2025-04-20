# domain/team.py
from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class TeamId:
    """Unique identifier for a team."""
    value: str

@dataclass
class TeamPerformance:
    """Performance metrics for a team."""
    avg_finish_position: float = 0.0
    avg_grid_position: float = 0.0
    avg_points_per_race: float = 0.0
    total_points: float = 0.0
    reliability_score: float = 0.0
    pit_stop_avg_time: float = 0.0

@dataclass
class Team:
    """Aggregate root for a team."""
    id: TeamId
    name: str
    performance: TeamPerformance
    
    @classmethod
    def create(
        cls, 
        team_id: str, 
        name: str, 
        performance_data: Optional[Dict] = None
    ) -> 'Team':
        """
        Factory method to create a team with optional performance data.
        
        Args:
            team_id: Unique identifier for the team
            name: Team name
            performance_data: Optional dictionary of performance metrics
            
        Returns:
            Team instance
        """
        team_id = TeamId(team_id)
        
        # Create default or use provided performance data
        perf_data = performance_data or {}
        performance = TeamPerformance(
            avg_finish_position=perf_data.get('avg_finish_position', 0.0),
            avg_grid_position=perf_data.get('avg_grid_position', 0.0),
            avg_points_per_race=perf_data.get('avg_points_per_race', 0.0),
            total_points=perf_data.get('total_points', 0.0),
            reliability_score=perf_data.get('reliability_score', 0.0),
            pit_stop_avg_time=perf_data.get('pit_stop_avg_time', 0.0)
        )
        
        return cls(
            id=team_id, 
            name=name, 
            performance=performance
        )
    
    def update_performance(self, race_result: Dict) -> None:
        """
        Update team's performance based on a race result.
        
        Args:
            race_result: Dictionary containing race performance data
        """
        # Implement logic to update performance metrics
        # This could include updating avg_finish_position, total_points, etc.
        pass
    
    def __repr__(self) -> str:
        return f"Team({self.name})"