"""VIGIL: Voice Integrity & Impersonation Guard

ML experimentation framework for speech deepfake detection.
"""

__version__ = "0.1.0"
__author__ = "VIGIL Team"

from vigil.detectors import get_detector, DeepfakeDetector

__all__ = ["get_detector", "DeepfakeDetector"]
