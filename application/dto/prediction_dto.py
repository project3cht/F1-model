# application/dto/prediction_dto.py
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class DriverDTO:
    """Data transfer object for driver information."""
    id: str
    name: str
    team: str
    is_rookie: bool = False

@dataclass
class PredictionResultDTO:
    """Data transfer object for individual prediction result."""
    driver: DriverDTO
    predicted_position: int
    interval_from_leader: float
    confidence_score: float = 0.0

@dataclass
class PredictionDTO:
    """Data transfer object for race prediction."""
    prediction_id: str
    race_id: str
    circuit_name: str
    timestamp: datetime
    safety_car_probability: float
    rain_probability: float
    results: List[PredictionResultDTO]
    
    @classmethod
    def from_domain(cls, prediction):
        """Create DTO from domain model."""
        results = []
        for result in prediction.results:
            driver_dto = DriverDTO(
                id=result.driver.id.value,
                name=str(result.driver.name),
                team=result.driver.team.name if hasattr(result.driver.team, 'name') else result.driver.team,
                is_rookie=result.driver.is_rookie
            )
            result_dto = PredictionResultDTO(
                driver=driver_dto,
                predicted_position=result.predicted_position,
                interval_from_leader=result.interval_from_leader or 0.0,
                confidence_score=result.confidence_score
            )
            results.append(result_dto)
            
        return cls(
            prediction_id=prediction.id.value,
            race_id=prediction.race.id.value,
            circuit_name=prediction.race.circuit.name,
            timestamp=prediction.timestamp,
            safety_car_probability=prediction.safety_car_probability,
            rain_probability=prediction.rain_probability,
            results=results
        )