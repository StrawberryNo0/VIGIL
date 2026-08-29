"""Abstract base classes for detectors and verifiers."""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class DeepfakeDetector(ABC):
    """Abstract interface for speech deepfake detection models.
    
    All detectors must implement this interface to be compatible with VIGIL.
    
    Note: Probabilities returned are model softmax outputs, NOT calibrated
    confidence scores. Use only for relative ranking unless calibration has
    been performed.
    """

    @abstractmethod
    def detect(self, audio_path: str) -> Dict[str, Any]:
        """Detect whether audio is synthetic or bonafide.
        
        Args:
            audio_path: Path to WAV file (16kHz PCM)
            
        Returns:
            {
                "synthetic_probability": float,   # 0.0-1.0, model softmax output
                "bonafide_probability": float,    # 0.0-1.0, model softmax output
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


class SpeakerVerifier(ABC):
    """Abstract interface for speaker verification models.
    
    WARNING: Speaker verification does NOT reliably detect synthetic speech.
    Use only for comparing speaker identity between two audio samples.
    Do NOT use the resulting embeddings to make synthetic-speech decisions.
    """

    @abstractmethod
    def embed(self, audio_path: str) -> Dict[str, Any]:
        """Generate speaker embedding for an audio file.
        
        Args:
            audio_path: Path to WAV file (16kHz PCM)
            
        Returns:
            {
                "embedding": np.ndarray,         # Speaker embedding vector
                "embedding_dim": int,            # Dimension of embedding
                "model_name": str,
                "latency_ms": float
            }
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio is invalid
            RuntimeError: If model inference fails
        """
        pass

    @abstractmethod
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute similarity between two embeddings.
        
        Args:
            embedding1: First speaker embedding
            embedding2: Second speaker embedding
            
        Returns:
            Similarity score (typically 0.0-1.0, cosine similarity)
            
        Raises:
            ValueError: If embeddings are invalid or incompatible
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
