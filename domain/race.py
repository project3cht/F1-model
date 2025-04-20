# domain/race.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime

from .driver import Driver
from .circuit import Circuit

@dataclass
class RaceId:
    """Unique identifier for a race."""
    value: str

@dataclass
class RaceResult:
    """Value object representing a driver's race result."""
    driver: Driver
    finishing_position: int
    points_scored: int
    interval_from_winner: Optional[float] = None
    grid_position: Optional[int] = None
    dnf: bool = False

@dataclass
class Race:
    """Aggregate root for a race event."""
    id: RaceId
    circuit: Circuit
    date: datetime
    season: int
    results: List[RaceResult] = field(default_factory=list)
    weather_conditions: Optional[Dict] = None
    safety_car_deployed: bool = False
    
    @classmethod
    def create(
        cls, 
        race_id: str, 
        circuit: Circuit, 
        date: datetime, 
        season: int,
        weather_conditions: Optional[Dict] = None
    ) -> 'Race':
        """
        Factory method to create a race event.
        
        Args:
            race_id: Unique identifier for the race
            circuit: Circuit where the race takes place
            date: Date of the race
            season: Racing season year
            weather_conditions: Optional weather information
            
        Returns:
            Race instance
        """
        race_id = RaceId(race_id)
        return cls(
            id=race_id,
            circuit=circuit,
            date=date,
            season=season,
            weather_conditions=weather_conditions
        )
    
    def add_race_result(
        self, 
        driver: Driver, 
        finishing_position: int, 
        points_scored: int,
        interval_from_winner: Optional[float] = None,
        grid_position: Optional[int] = None,
        dnf: bool = False
    ) -> None:
        """
        Add a driver's race result to the race.
        
        Args:
            driver: Driver who participated in the race
            finishing_position: Final race position
            points_scored: Points earned in the race
            interval_from_winner: Time interval from the race winner
            grid_position: Starting grid position
            dnf: Did Not Finish flag
        """
        result = RaceResult(
            driver=driver,
            finishing_position=finishing_position,
            points_scored=points_scored,
            interval_from_winner=interval_from_winner,
            grid_position=grid_position,
            dnf=dnf
        )
        self.results.append(result)
    
    def get_winner(self) -> Optional[RaceResult]:
        """
        Get the race winner.
        
        Returns:
            RaceResult of the winning driver, or None if no winner
        """
        winners = [result for result in self.results if result.finishing_position == 1]
        return winners[0] if winners else None
    
    def calculate_championship_points(self) -> Dict[Driver, int]:
        """
        Calculate championship points for the race.
        
        Returns:
            Dictionary of drivers and their earned championship points
        """
        return {result.driver: result.points_scored for result in self.results}
    
    def __repr__(self) -> str:
        return f"Race(id={self.id}, circuit={self.circuit}, date={self.date})"