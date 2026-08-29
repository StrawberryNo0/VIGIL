"""Audio loading and preprocessing utilities."""

import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import librosa
from scipy.io import wavfile

logger = logging.getLogger(__name__)


def load_audio(
    audio_path: str,
    target_sr: int = 16000,
    mono: bool = True,
    max_duration: Optional[float] = None
) -> Tuple[Optional[np.ndarray], int]:
    """Load audio file with resampling and validation.
    
    Args:
        audio_path: Path to audio file (WAV, MP3, FLAC, etc.)
        target_sr: Target sample rate (Hz)
        mono: Convert to mono if True
        max_duration: Maximum duration in seconds (clips if exceeded)
        
    Returns:
        Tuple of (waveform, sample_rate) or (None, None) on error
        
    Warnings:
        - Extremely short audio (<0.5s) logged as warning
        - Resampling applied if sample rate doesn't match target
        - Mono conversion applied if stereo
    """
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return None, None
    
    if audio_path.suffix.lower() not in [".wav", ".mp3", ".flac", ".ogg"]:
        logger.error(f"Unsupported audio format: {audio_path.suffix}")
        return None, None
    
    try:
        # Load audio with librosa (handles multiple formats)
        waveform, sr = librosa.load(
            str(audio_path),
            sr=target_sr if target_sr else None,
            mono=mono,
            duration=max_duration
        )
        
        # Validate
        if len(waveform) == 0:
            logger.error(f"Empty audio: {audio_path}")
            return None, None
        
        duration_s = len(waveform) / target_sr
        if duration_s < 0.5:
            logger.warning(f"Very short audio ({duration_s:.2f}s): {audio_path}")
        
        logger.debug(f"Loaded {audio_path}: {duration_s:.2f}s @ {target_sr}Hz, shape={waveform.shape}")
        return waveform, target_sr
    
    except Exception as e:
        logger.error(f"Error loading {audio_path}: {str(e)}")
        return None, None


def validate_audio(waveform: np.ndarray, sr: int, min_duration_s: float = 0.1) -> bool:
    """Validate audio waveform.
    
    Args:
        waveform: Audio waveform array
        sr: Sample rate in Hz
        min_duration_s: Minimum duration in seconds
        
    Returns:
        True if audio is valid, False otherwise
    """
    if waveform is None or len(waveform) == 0:
        return False
    
    duration = len(waveform) / sr
    if duration < min_duration_s:
        logger.warning(f"Audio duration {duration:.2f}s below minimum {min_duration_s}s")
        return False
    
    # Check for NaN/Inf
    if np.any(~np.isfinite(waveform)):
        logger.error("Audio contains NaN or Inf values")
        return False
    
    return True
