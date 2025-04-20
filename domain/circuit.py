# domain/circuit.py
from dataclasses import dataclass, field
from typing import Optional, Dict, List

@dataclass
class CircuitId:
    """Unique identifier for a circuit."""
    value: str

@dataclass
class CircuitCharacteristics:
    """Value object representing circuit-specific characteristics."""
    length_km: float
    lap_record: Optional[float] = None
    corner_count: int = 0
    track_type: str = "Unknown"  # e.g., street, power, technical, balanced
    
    def is_challenging(self) -> bool:
        """
        Determine if the circuit is considered challenging.
        
        Returns:
            bool: True if the circuit is considered technically demanding
        """
        return (self.corner_count > 12 or 
                self.track_type in ["street", "technical"])

@dataclass
class WeatherConditions:
    """Value object representing circuit weather characteristics."""
    typical_temperature: Optional[float] = None
    rainfall_probability: float = 0.0
    wind_speed: Optional[float] = None
    
    def is_likely_wet(self) -> bool:
        """
        Determine if the circuit is likely to have wet conditions.
        
        Returns:
            bool: True if rainfall probability is high
        """
        return self.rainfall_probability > 0.5

@dataclass
class Circuit:
    """Aggregate root for a racing circuit."""
    id: CircuitId
    name: str
    country: str
    city: Optional[str] = None
    characteristics: Optional[CircuitCharacteristics] = None
    typical_weather: Optional[WeatherConditions] = None
    historical_race_data: List[Dict] = field(default_factory=list)
    
    @classmethod
    def create(
        cls, 
        circuit_id: str, 
        name: str, 
        country: str,
        city: Optional[str] = None,
        length_km: Optional[float] = None,
        track_type: Optional[str] = None,
        corner_count: int = 0
    ) -> 'Circuit':
        """
        Factory method to create a circuit.
        
        Args:
            circuit_id: Unique identifier for the circuit
            name: Circuit name
            country: Country where the circuit is located
            city: Optional city where the circuit is located
            length_km: Optional circuit length
            track_type: Optional track type
            corner_count: Number of corners
            
        Returns:
            Circuit instance
        """
        circuit_id = CircuitId(circuit_id)
        
        # Create circuit characteristics if length or track type provided
        characteristics = None
        if length_km or track_type or corner_count:
            characteristics = CircuitCharacteristics(
                length_km=length_km or 0.0,
                track_type=track_type or "Unknown",
                corner_count=corner_count
            )
        
        return cls(
            id=circuit_id,
            name=name,
            country=country,
            city=city,
            characteristics=characteristics
        )
    
    def add_historical_race_data(self, race_data: Dict) -> None:
        """
        Add historical race data for the circuit.
        
        Args:
            race_data: Dictionary containing race performance data
        """
        self.historical_race_data.append(race_data)
    
    def get_historical_winners(self) -> List[str]:
        """
        Get list of historical race winners at this circuit.
        
        Returns:
            List of driver names who have won at this circuit
        """
        return [
            race.get('winner', '') 
            for race in self.historical_race_data 
            if 'winner' in race
        ]
    
    def is_challenging_circuit(self) -> bool:
        """
        Determine if the circuit is considered challenging.
        
        Returns:
            bool: True if the circuit is technically demanding
        """
        return (self.characteristics is not None and 
                self.characteristics.is_challenging())
    
    def __repr__(self) -> str:
        return f"Circuit({self.name}, {self.country})"