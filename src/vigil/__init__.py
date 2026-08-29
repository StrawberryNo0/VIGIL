"""VIGIL: Voice Integrity & Impersonation Guard

ML experimentation framework for speech deepfake detection.
"""

__version__ = "0.2.0"
__author__ = "VIGIL Team"

from vigil.detectors import get_detector, get_verifier, DeepfakeDetector, SpeakerVerifier

__all__ = ["get_detector", "get_verifier", "DeepfakeDetector", "SpeakerVerifier"]
