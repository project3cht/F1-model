# domain/driver.py
from dataclasses import dataclass, field
from typing import Optional, Dict, TYPE_CHECKING

# Use TYPE_CHECKING to handle circular imports
if TYPE_CHECKING:
    from domain.team import Team

@dataclass
class DriverId:
    """Unique identifier for a driver."""
    value: str

@dataclass
class DriverName:
    """Value object representing a driver's name."""
    first_name: str
    last_name: str
    
    def __str__(self) -> str:
        return f"{self.first_name} {self.last_name}"

@dataclass
class DriverPerformance:
    """Performance metrics for a driver."""
    avg_finish_position: float = 0.0
    avg_grid_position: float = 0.0
    positions_gained: float = 0.0
    finishing_rate: float = 0.0
    win_rate: float = 0.0
    podium_rate: float = 0.0

@dataclass
class Driver:
    """Aggregate root for a driver."""
    id: DriverId
    name: DriverName
    team: 'Team'  # Forward reference to avoid circular import
    performance: DriverPerformance = field(default_factory=DriverPerformance)
    is_rookie: bool = False

    @classmethod
    def create(
        cls, 
        driver_id: str, 
        first_name: str, 
        last_name: str, 
        team: 'Team', 
        performance_data: Optional[Dict] = None,
        is_rookie: bool = False
    ) -> 'Driver':
        """
        Factory method to create a driver with optional performance data.
        
        Args:
            driver_id: Unique identifier for the driver
            first_name: Driver's first name
            last_name: Driver's last name
            team: Driver's team
            performance_data: Optional dictionary of performance metrics
            is_rookie: Whether the driver is a rookie
            
        Returns:
            Driver instance
        """
        driver_id = DriverId(driver_id)
        name = DriverName(first_name, last_name)
        
        # Create default or use provided performance data
        perf_data = performance_data or {}
        performance = DriverPerformance(
            avg_finish_position=perf_data.get('avg_finish_position', 0.0),
            avg_grid_position=perf_data.get('avg_grid_position', 0.0),
            positions_gained=perf_data.get('positions_gained', 0.0),
            finishing_rate=perf_data.get('finishing_rate', 0.0),
            win_rate=perf_data.get('win_rate', 0.0),
            podium_rate=perf_data.get('podium_rate', 0.0)
        )
        
        return cls(
            id=driver_id, 
            name=name, 
            team=team, 
            performance=performance,
            is_rookie=is_rookie
        )

    def update_performance(self, race_result: Dict) -> None:
        """
        Update driver's performance based on a race result.
        
        Args:
            race_result: Dictionary containing race performance data
        """
        # Implement logic to update performance metrics
        # This could include updating avg_finish_position, positions_gained, etc.
        if 'position' in race_result:
            # Update average finish position (running average)
            total_races = 1  # Default to 1 if not specified
            if hasattr(self, '_total_races'):
                total_races = self._total_races + 1
            
            self.performance.avg_finish_position = (
                (self.performance.avg_finish_position * (total_races - 1) + 
                 race_result['position']) / total_races
            )
            
            # Update positions gained
            if 'grid_position' in race_result:
                positions_gained = race_result['grid_position'] - race_result['position']
                self.performance.positions_gained = (
                    (self.performance.positions_gained * (total_races - 1) + 
                     positions_gained) / total_races
                )
            
            # Update finishing statistics
            if race_result.get('finished', True):
                self.performance.finishing_rate = (
                    (self.performance.finishing_rate * (total_races - 1) + 1) / total_races
                )
            
            # Update win and podium rates
            if race_result['position'] == 1:
                self.performance.win_rate = (
                    (self.performance.win_rate * (total_races - 1) + 1) / total_races
                )
            elif race_result['position'] <= 3:
                self.performance.podium_rate = (
                    (self.performance.podium_rate * (total_races - 1) + 1) / total_races
                )
            
            # Track total races
            self._total_races = total_races

    def __repr__(self) -> str:
        return f"Driver({self.name}, Team: {self.team})"