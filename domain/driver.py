# domain/driver.py
from dataclasses import dataclass
from typing import Optional, Dict

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
    performance: DriverPerformance
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
        pass
    
    def __repr__(self) -> str:
        return f"Driver({self.name}, Team: {self.team})"