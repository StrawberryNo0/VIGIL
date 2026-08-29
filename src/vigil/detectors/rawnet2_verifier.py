"""RawNet2-based speaker verification model."""

import logging
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from transformers import AutoModel, AutoFeatureExtractor

from vigil.detectors.base import SpeakerVerifier
from vigil.utils.audio import load_audio

logger = logging.getLogger(__name__)


class RawNet2Verifier(SpeakerVerifier):
    """RawNet2-based speaker verification using embeddings.
    
    Uses RawNet2 architecture for raw waveform processing.
    Generates speaker embeddings for speaker identification/verification.
    
    WARNING: This is a speaker verification model, NOT a synthetic-speech detector.
    Embeddings can be used to determine if two speakers are the same person,
    but cannot reliably distinguish synthetic from bonafide speech.
    
    Supports 16kHz PCM WAV files.
    """

    def __init__(self, model_path: str = None, device: str = "cpu"):
        """Initialize the verifier.
        
        Args:
            model_path: Optional path to custom model. Defaults to HuggingFace model.
            device: "cpu" or "cuda"
        """
        self._device = device
        self._model_path = model_path or "ashraq/rawnet2-speaker-verification"
        
        logger.info(f"Loading RawNet2 model from {self._model_path}")
        self.model = AutoModel.from_pretrained(
            self._model_path,
            trust_remote_code=True,
            cache_dir=None
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self._model_path,
            trust_remote_code=True,
            cache_dir=None
        )
        
        self.model.to(device)
        self.model.eval()
        logger.info(f"Model loaded on device: {device}")

    def embed(self, audio_path: str) -> Dict[str, Any]:
        """Generate speaker embedding for an audio file.
        
        Args:
            audio_path: Path to 16kHz PCM WAV file
            
        Returns:
            Speaker embedding and metadata
            
        Raises:
            FileNotFoundError: If audio file not found
            ValueError: If audio is invalid
            RuntimeError: If model inference fails
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            # Load audio
            waveform, sr = load_audio(str(audio_path), target_sr=16000)
            if waveform is None or len(waveform) == 0:
                raise ValueError(f"Failed to load audio: {audio_path}")

            # Prepare input
            inputs = self.feature_extractor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Inference - generate embedding
            start_time = time.time()
            with torch.no_grad():
                embedding_output = self.model(**inputs).last_hidden_state
            latency_ms = (time.time() - start_time) * 1000.0

            # Pool and normalize embedding
            embedding = embedding_output.mean(dim=1)[0].cpu().numpy()
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

            result = {
                "embedding": embedding,
                "embedding_dim": int(embedding.shape[0]),
                "model_name": "rawnet2",
                "latency_ms": latency_ms
            }

            logger.debug(f"Embedding result: dim={result['embedding_dim']}, latency={latency_ms:.1f}ms")
            return result

        except Exception as e:
            logger.error(f"Error during embedding on {audio_path}: {str(e)}")
            raise RuntimeError(f"Embedding failed for {audio_path}: {str(e)}") from e

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two speaker embeddings.
        
        Args:
            embedding1: First speaker embedding
            embedding2: Second speaker embedding
            
        Returns:
            Cosine similarity (0.0-1.0, where 1.0 means identical speakers)
            
        Raises:
            ValueError: If embeddings are invalid or incompatible
        """
        if embedding1 is None or embedding2 is None:
            raise ValueError("Both embeddings must be non-None")
        
        if not isinstance(embedding1, np.ndarray) or not isinstance(embedding2, np.ndarray):
            raise ValueError("Embeddings must be numpy arrays")
        
        if embedding1.shape != embedding2.shape:
            raise ValueError(
                f"Embeddings must have same shape. Got {embedding1.shape} and {embedding2.shape}"
            )
        
        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            raise ValueError("Cannot compute similarity for zero-norm embedding")
        
        similarity = dot_product / (norm1 * norm2)
        
        # Clamp to [0, 1] in case of numerical issues
        similarity = float(np.clip(similarity, 0.0, 1.0))
        
        logger.debug(f"Similarity: {similarity:.4f}")
        return similarity

    @property
    def model_name(self) -> str:
        """Return model name."""
        return "rawnet2"

    @property
    def device(self) -> str:
        """Return device."""
        return self._device
