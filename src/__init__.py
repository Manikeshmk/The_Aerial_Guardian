"""
The Aerial Guardian - Drone-based Person Detection and Tracking Pipeline
Optimized for small-scale object detection and ID maintenance in drone footage.
"""

from .detector import AerialDetector
from .tracker import DroneByteTrac
from .pipeline import DroneVideoProcessor, AerialPreprocessor

__version__ = "1.0.0"
__author__ = "Manikesh"

__all__ = [
    'AerialDetector',
    'DroneByteTrac',
    'DroneVideoProcessor',
    'AerialPreprocessor',
]
