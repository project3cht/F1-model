# domain/prediction.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime

from .race import Race, RaceResult
from .driver import Driver
from .circuit import Circuit

@dataclass
class PredictionId:
    """Unique identifier for a prediction."""
    value: str

@dataclass
class PredictionResult:
    """Represents the predicted result for a single driver."""
    driver: Driver
    predicted_position: float
    probability: float = 1.0
    interval_from_leader: Optional[float] = None
    confidence_score: float = 0.0

@dataclass
class Prediction:
    """Aggregate root for race predictions."""
    id: PredictionId
    race: Race
    timestamp: datetime
    results: List[PredictionResult] = field(default_factory=list)
    
    # Prediction parameters
    safety_car_probability: float = 0.6
    rain_probability: float = 0.0
    
    @classmethod
    def create(
        cls, 
        prediction_id: str, 
        race: Race,
        safety_car_probability: float = 0.6,
        rain_probability: float = 0.0
    ) -> 'Prediction':
        """
        Factory method to create a prediction.
        
        Args:
            prediction_id: Unique identifier for the prediction
            race: Race being predicted
            safety_car_probability: Probability of safety car
            rain_probability: Probability of rain
            
        Returns:
            Prediction instance
        """
        prediction_id = PredictionId(prediction_id)
        return cls(
            id=prediction_id,
            race=race,
            timestamp=datetime.now(),
            safety_car_probability=safety_car_probability,
            rain_probability=rain_probability
        )
    
    def add_prediction_result(
        self, 
        driver: Driver, 
        predicted_position: float,
        probability: float = 1.0,
        interval_from_leader: Optional[float] = None,
        confidence_score: float = 0.0
    ) -> None:
        """
        Add a driver's predicted result.
        
        Args:
            driver: Driver being predicted
            predicted_position: Estimated finishing position
            probability: Probability of this prediction
            interval_from_leader: Time interval from race leader
            confidence_score: Confidence in the prediction
        """
        result = PredictionResult(
            driver=driver,
            predicted_position=predicted_position,
            probability=probability,
            interval_from_leader=interval_from_leader,
            confidence_score=confidence_score
        )
        self.results.append(result)
    
    def get_predicted_winner(self) -> Optional[PredictionResult]:
        """
        Get the predicted race winner.
        
        Returns:
            PredictionResult of the predicted winner, or None
        """
        winners = [result for result in self.results if result.predicted_position == 1.0]
        return winners[0] if winners else None
    
    def sort_predictions(self) -> List[PredictionResult]:
        """
        Sort predictions by predicted position.
        
        Returns:
            Sorted list of prediction results
        """
        return sorted(self.results, key=lambda x: x.predicted_position)
    
    def get_confidence_summary(self) -> Dict[str, float]:
        """
        Get an overall confidence summary of the prediction.
        
        Returns:
            Dictionary with prediction confidence metrics
        """
        return {
            'avg_confidence': sum(r.confidence_score for r in self.results) / len(self.results),
            'max_confidence': max(r.confidence_score for r in self.results),
            'min_confidence': min(r.confidence_score for r in self.results)
        }
    
    def __repr__(self) -> str:
        return f"Prediction(race={self.race}, timestamp={self.timestamp})"