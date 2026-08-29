"""Wav2Vec2-based speech spoofing detector."""

import logging
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoProcessor

from vigil.detectors.base import DeepfakeDetector
from vigil.utils.audio import load_audio

logger = logging.getLogger(__name__)


class Wav2Vec2Detector(DeepfakeDetector):
    """Wav2Vec2 multilingual spoofing detection model.
    
    Uses HuggingFace's wav2vec2-xlsr fine-tuned for speech spoofing detection.
    Supports 16kHz PCM WAV files.
    """

    def __init__(self, model_path: str = None, device: str = "cpu"):
        """Initialize the detector.
        
        Args:
            model_path: Optional path to custom model. Defaults to HuggingFace model.
            device: "cpu" or "cuda"
        """
        self._device = device
        self._model_path = model_path or "aniemore/wav2vec2-xlsr-multilingual-speech-spoofing-detection"
        
        logger.info(f"Loading Wav2Vec2 model from {self._model_path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self._model_path,
            trust_remote_code=True,
            cache_dir=None
        )
        self.processor = AutoProcessor.from_pretrained(
            self._model_path,
            trust_remote_code=True,
            cache_dir=None
        )
        
        self.model.to(device)
        self.model.eval()
        logger.info(f"Model loaded on device: {device}")

    def detect(self, audio_path: str) -> Dict[str, Any]:
        """Detect synthetic vs bonafide speech.
        
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
            inputs = self.processor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Inference
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(**inputs)
            latency_ms = (time.time() - start_time) * 1000.0

            # Parse output
            logits = outputs.logits[0].cpu().numpy()
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()

            # Assuming class 0=bonafide, 1=synthetic (common in spoofing models)
            bonafide_prob = float(probs[0])
            synthetic_prob = float(probs[1])

            result = {
                "synthetic_probability": synthetic_prob,
                "bonafide_probability": bonafide_prob,
                "model_name": "wav2vec2-spoofing",
                "latency_ms": latency_ms
            }

            logger.debug(f"Detection result: {result}")
            return result

        except Exception as e:
            logger.error(f"Error during detection on {audio_path}: {str(e)}")
            raise RuntimeError(f"Detection failed for {audio_path}: {str(e)}") from e

    @property
    def model_name(self) -> str:
        """Return model name."""
        return "wav2vec2-spoofing"

    @property
    def device(self) -> str:
        """Return device."""
        return self._device
