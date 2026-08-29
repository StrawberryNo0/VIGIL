"""Speech deepfake detection module."""

from vigil.detectors.base import DeepfakeDetector
from vigil.detectors.wav2vec2_detector import Wav2Vec2Detector
from vigil.detectors.rawnet2_detector import RawNet2Detector


def get_detector(model_name: str, model_path: str = None, device: str = "cpu") -> DeepfakeDetector:
    """Factory function to instantiate a detector.
    
    Args:
        model_name: Name of the detector ("wav2vec2-spoofing" or "rawnet2")
        model_path: Optional custom model path
        device: "cpu" or "cuda"
        
    Returns:
        DeepfakeDetector instance
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name == "wav2vec2-spoofing":
        return Wav2Vec2Detector(model_path=model_path, device=device)
    elif model_name == "rawnet2":
        return RawNet2Detector(model_path=model_path, device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: wav2vec2-spoofing, rawnet2")


__all__ = ["DeepfakeDetector", "get_detector", "Wav2Vec2Detector", "RawNet2Detector"]
