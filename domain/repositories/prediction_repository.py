# domain/repositories/prediction_repository.py
from abc import ABC, abstractmethod
from typing import Optional, List

from ..prediction import Prediction, PredictionId
from ..race import Race

class PredictionRepository(ABC):
    """
    Abstract base class for prediction repository.
    Defines the contract for storing and retrieving predictions.
    """
    
    @abstractmethod
    def save(self, prediction: Prediction) -> None:
        """
        Save a prediction to the repository.
        
        Args:
            prediction: Prediction to be saved
        """
        pass
    
    @abstractmethod
    def find_by_id(self, prediction_id: PredictionId) -> Optional[Prediction]:
        """
        Find a prediction by its ID.
        
        Args:
            prediction_id: ID of the prediction to find
        
        Returns:
            Prediction if found, None otherwise
        """
        pass
    
    @abstractmethod
    def find_by_race(self, race: Race) -> List[Prediction]:
        """
        Find predictions for a specific race.
        
        Args:
            race: Race to find predictions for
        
        Returns:
            List of predictions for the race
        """
        pass
    
    @abstractmethod
    def delete(self, prediction: Prediction) -> None:
        """
        Delete a prediction from the repository.
        
        Args:
            prediction: Prediction to be deleted
        """
        pass

# domain/services/prediction_service.py
from typing import List, Optional, Dict
from uuid import uuid4

from ..race import Race
from ..driver import Driver
from ..prediction import Prediction
from .prediction_repository import PredictionRepository

class PredictionService:
    """
    Domain service for managing race predictions.
    """
    
    def __init__(self, prediction_repository: PredictionRepository):
        """
        Initialize prediction service with a repository.
        
        Args:
            prediction_repository: Repository for storing predictions
        """
        self._repository = prediction_repository
    
    def create_prediction(
        self, 
        race: Race, 
        prediction_parameters: Optional[Dict] = None
    ) -> Prediction:
        """
        Create a new prediction for a race.
        
        Args:
            race: Race to create prediction for
            prediction_parameters: Optional parameters affecting prediction
            
        Returns:
            Created Prediction
        """
        # Generate unique prediction ID
        prediction_id = str(uuid4())
        
        # Default parameters if not provided
        params = prediction_parameters or {}
        safety_car_prob = params.get('safety_car_probability', 0.6)
        rain_prob = params.get('rain_probability', 0.0)
        
        # Create prediction
        prediction = Prediction.create(
            prediction_id, 
            race, 
            safety_car_probability=safety_car_prob,
            rain_probability=rain_prob
        )
        
        # Save prediction to repository
        self._repository.save(prediction)
        
        return prediction
    
    def generate_prediction_results(
        self, 
        prediction: Prediction, 
        drivers: List[Driver]
    ) -> Prediction:
        """
        Generate prediction results for a set of drivers.
        
        Args:
            prediction: Prediction to add results to
            drivers: List of drivers to predict
            
        Returns:
            Updated Prediction with results
        """
        # This method would typically involve complex prediction logic
        # For now, a simple placeholder implementation
        for i, driver in enumerate(drivers, 1):
            predicted_position = float(i)
            
            prediction.add_prediction_result(
                driver=driver,
                predicted_position=predicted_position,
                probability=0.8,  # Example probability
                interval_from_leader=None,
                confidence_score=0.7  # Example confidence
            )
        
        # Save updated prediction
        self._repository.save(prediction)
        
        return prediction
    
    def get_prediction_confidence(self, prediction: Prediction) -> Dict[str, float]:
        """
        Get confidence metrics for a prediction.
        
        Args:
            prediction: Prediction to analyze
            
        Returns:
            Dictionary of confidence metrics
        """
        return prediction.get_confidence_summary()
    
    def compare_predictions(
        self, 
        prediction1: Prediction, 
        prediction2: Prediction
    ) -> Dict[str, any]:
        """
        Compare two predictions.
        
        Args:
            prediction1: First prediction to compare
            prediction2: Second prediction to compare
            
        Returns:
            Comparison metrics
        """
        # Implement prediction comparison logic
        results1 = prediction1.sort_predictions()
        results2 = prediction2.sort_predictions()
        
        # Example comparison metrics
        return {
            'total_differences': sum(
                1 for p1, p2 in zip(results1, results2) 
                if abs(p1.predicted_position - p2.predicted_position) > 0.5
            ),
            'avg_position_delta': sum(
                abs(p1.predicted_position - p2.predicted_position) 
                for p1, p2 in zip(results1, results2)
            ) / len(results1)
        }