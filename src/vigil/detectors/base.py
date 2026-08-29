"""Abstract base class for deepfake detectors."""

from abc import ABC, abstractmethod
from typing import Dict, Any


class DeepfakeDetector(ABC):
    """Abstract interface for speech deepfake detection models.
    
    All detectors must implement this interface to be compatible with VIGIL.
    """

    @abstractmethod
    def detect(self, audio_path: str) -> Dict[str, Any]:
        """Detect whether audio is synthetic or bonafide.
        
        Args:
            audio_path: Path to WAV file (16kHz PCM)
            
        Returns:
            {
                "synthetic_probability": float,   # 0.0-1.0
                "bonafide_probability": float,    # 0.0-1.0
                "model_name": str,
                "latency_ms": float
            }
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio is invalid
            RuntimeError: If model inference fails
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model."""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Return the device (cpu or cuda)."""
        pass
