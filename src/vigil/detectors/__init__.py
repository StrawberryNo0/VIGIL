"""Speech deepfake detection and speaker verification module."""

from vigil.detectors.base import DeepfakeDetector, SpeakerVerifier
from vigil.detectors.wav2vec2_detector import Wav2Vec2Detector
from vigil.detectors.rawnet2_verifier import RawNet2Verifier


def get_detector(model_name: str, model_path: str = None, device: str = "cpu") -> DeepfakeDetector:
    """Factory function to instantiate a deepfake detector.
    
    Args:
        model_name: Name of the detector ("wav2vec2-spoofing")
        model_path: Optional custom model path
        device: "cpu" or "cuda"
        
    Returns:
        DeepfakeDetector instance
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name == "wav2vec2-spoofing":
        return Wav2Vec2Detector(model_path=model_path, device=device)
    else:
        raise ValueError(
            f"Unknown detector: {model_name}. "
            f"Available detectors: wav2vec2-spoofing"
        )


def get_verifier(model_name: str, model_path: str = None, device: str = "cpu") -> SpeakerVerifier:
    """Factory function to instantiate a speaker verifier.
    
    Args:
        model_name: Name of the verifier ("rawnet2")
        model_path: Optional custom model path
        device: "cpu" or "cuda"
        
    Returns:
        SpeakerVerifier instance
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name == "rawnet2":
        return RawNet2Verifier(model_path=model_path, device=device)
    else:
        raise ValueError(
            f"Unknown verifier: {model_name}. "
            f"Available verifiers: rawnet2"
        )


__all__ = [
    "DeepfakeDetector",
    "SpeakerVerifier",
    "get_detector",
    "get_verifier",
    "Wav2Vec2Detector",
    "RawNet2Verifier",
]
