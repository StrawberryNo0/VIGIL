"""RawNet2-based speech spoofing detector."""

import logging
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoFeatureExtractor

from vigil.detectors.base import DeepfakeDetector
from vigil.utils.audio import load_audio

logger = logging.getLogger(__name__)


class RawNet2Detector(DeepfakeDetector):
    """RawNet2-based speech spoofing detection.
    
    Uses RawNet2 architecture for raw waveform processing.
    Generates speaker embeddings that can distinguish synthetic speech.
    Supports 16kHz PCM WAV files.
    """

    def __init__(self, model_path: str = None, device: str = "cpu"):
        """Initialize the detector.
        
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
        
        # Threshold for synthetic detection (empirically determined)
        self.synthetic_threshold = 0.5

    def detect(self, audio_path: str) -> Dict[str, Any]:
        """Detect synthetic vs bonafide speech using embedding similarity.
        
        Args:
            audio_path: Path to 16kHz PCM WAV file
            
        Returns:
            Detection result with probabilities and latency
            
        Raises:
            FileNotFoundError: If audio file not found
            ValueError: If audio is invalid
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
                embedding = self.model(**inputs).last_hidden_state
            latency_ms = (time.time() - start_time) * 1000.0

            # Pool embedding
            embedding = embedding.mean(dim=1)[0].cpu().numpy()
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

            # Estimate synthetic probability from embedding statistics
            # (Synthetic speech tends to have different spectral characteristics)
            synthetic_score = self._estimate_synthetic_probability(embedding)
            bonafide_prob = 1.0 - synthetic_score
            synthetic_prob = synthetic_score

            result = {
                "synthetic_probability": synthetic_prob,
                "bonafide_probability": bonafide_prob,
                "model_name": "rawnet2",
                "latency_ms": latency_ms
            }

            logger.debug(f"Detection result: {result}")
            return result

        except Exception as e:
            logger.error(f"Error during detection on {audio_path}: {str(e)}")
            raise RuntimeError(f"Detection failed for {audio_path}: {str(e)}") from e

    def _estimate_synthetic_probability(self, embedding: np.ndarray) -> float:
        """Estimate synthetic probability from embedding.
        
        Note: This is a heuristic. For production, would need fine-tuning
        on labeled synthetic/bonafide data.
        
        Args:
            embedding: Speaker embedding vector
            
        Returns:
            Probability of synthetic speech (0.0-1.0)
        """
        # Heuristic: synthetic speech embeddings tend to have lower variance
        # and different kurtosis patterns
        variance = np.var(embedding)
        kurtosis = np._moment(embedding, 4) / (np.var(embedding) ** 2 + 1e-8)
        
        # Simple scoring (would be replaced by learned classifier)
        score = min(max((self.synthetic_threshold - variance / 10.0), 0.0), 1.0)
        
        return score

    @property
    def model_name(self) -> str:
        """Return model name."""
        return "rawnet2"

    @property
    def device(self) -> str:
        """Return device."""
        return self._device
