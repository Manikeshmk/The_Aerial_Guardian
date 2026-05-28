"""
Aerial imagery preprocessing optimized for small object detection.
Handles drone-specific challenges: low resolution, high altitude, motion blur.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import torch
import torchvision.transforms as transforms

class AerialPreprocessor:
    """
    Preprocessing pipeline for drone footage.
    Optimizations:
    - Histogram equalization for low-light scenes
    - Contrast enhancement for small objects
    - Super-resolution for upsampling
    - Motion blur reduction
    """
    
    def __init__(self, enable_clahe: bool = True, enable_sharpening: bool = True):
        """
        Args:
            enable_clahe: Use CLAHE for contrast enhancement
            enable_sharpening: Apply sharpening to enhance edges
        """
        self.enable_clahe = enable_clahe
        self.enable_sharpening = enable_sharpening
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
    def preprocess(self, frame: np.ndarray, denoise: bool = False) -> np.ndarray:
        """
        Apply preprocessing pipeline to frame.
        
        Args:
            frame: Input BGR image
            denoise: Apply denoising
            
        Returns:
            Preprocessed image
        """
        # Denoise if needed (use bilateral filter for edge-preserving denoising)
        if denoise:
            frame = cv2.bilateralFilter(frame, 5, 75, 75)
        
        # Apply CLAHE for contrast enhancement
        if self.enable_clahe:
            # Convert to LAB for better contrast enhancement
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        # Sharpening
        if self.enable_sharpening:
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]]) / 1.0
            frame = cv2.filter2D(frame, -1, kernel)
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        return frame
    
    @staticmethod
    def reduce_motion_blur(frame: np.ndarray) -> np.ndarray:
        """
        Reduce motion blur using Wiener filtering.
        For drone footage with significant motion.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Estimate blur using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # If blur is significant, apply sharpening
        if laplacian_var < 100:
            # Apply unsharp masking
            gaussian = cv2.GaussianBlur(frame, (0, 0), 1.0)
            frame = cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        return frame
    
    @staticmethod
    def adaptive_resize(frame: np.ndarray, target_height: int = 720) -> Tuple[np.ndarray, float]:
        """
        Adaptively resize frame while preserving aspect ratio.
        Returns scale factor for coordinate transformation.
        
        Args:
            frame: Input image
            target_height: Target height
            
        Returns:
            Resized frame and scale factor
        """
        h, w = frame.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        
        if scale < 1:
            resized = cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
        
        return resized, scale


class DroneVideoProcessor:
    """
    Main video processing pipeline for drone footage analysis.
    Integrates detection, tracking, and visualization.
    """
    
    def __init__(self, detector, tracker, preprocessor: Optional[AerialPreprocessor] = None):
        """
        Args:
            detector: Instance of AerialDetector
            tracker: Instance of DroneByteTrac
            preprocessor: Instance of AerialPreprocessor
        """
        self.detector = detector
        self.tracker = tracker
        self.preprocessor = preprocessor or AerialPreprocessor()
        
        self.frame_count = 0
        self.fps_times = []
        self.detection_times = []
        self.tracking_times = []
        
    def process_frame(self, frame: np.ndarray, apply_preprocessing: bool = True) -> Tuple[np.ndarray, list]:
        """
        Process single frame: detect and track persons.
        
        Args:
            frame: Input BGR image
            apply_preprocessing: Apply preprocessing pipeline
            
        Returns:
            Annotated frame and tracking results
        """
        import time
        
        self.frame_count += 1
        frame_start = time.time()
        
        # Preprocessing
        if apply_preprocessing:
            frame_proc = self.preprocessor.preprocess(frame.copy(), denoise=False)
        else:
            frame_proc = frame.copy()
        
        # Detection
        det_start = time.time()
        detections = self.detector.detect_with_nms(frame_proc, target_size=640)
        det_time = time.time() - det_start
        self.detection_times.append(det_time)
        
        # Tracking
        track_start = time.time()
        tracks = self.tracker.update(detections, frame)
        track_time = time.time() - track_start
        self.tracking_times.append(track_time)
        
        # Visualization
        frame_vis = self._draw_results(frame, tracks)
        
        frame_total = time.time() - frame_start
        self.fps_times.append(frame_total)
        
        return frame_vis, tracks
    
    @staticmethod
    def _draw_results(frame: np.ndarray, tracks: list) -> np.ndarray:
        """Draw bounding boxes and tracking info on frame."""
        frame = frame.copy()
        
        for track in tracks:
            track_id = track['track_id']
            x1, y1, x2, y2 = map(int, track['bbox'])
            trajectory = track['trajectory']
            
            # Draw bounding box
            color = AerialDetector._id_to_color(track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID
            label = f"ID: {track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
            
            # Draw trajectory
            if len(trajectory) > 1:
                points = np.array(trajectory, dtype=np.int32)
                cv2.polylines(frame, [points], False, color, 1)
                # Draw trajectory endpoints
                for pt in points[-3:]:
                    cv2.circle(frame, tuple(pt), 2, color, -1)
        
        return frame
    
    def get_fps(self) -> float:
        """Get average FPS."""
        if len(self.fps_times) == 0:
            return 0.0
        return 1.0 / np.mean(self.fps_times)
    
    def get_statistics(self) -> dict:
        """Get processing statistics."""
        stats = {
            'total_frames': self.frame_count,
            'avg_fps': self.get_fps(),
            'avg_detection_time': np.mean(self.detection_times) if self.detection_times else 0,
            'avg_tracking_time': np.mean(self.tracking_times) if self.tracking_times else 0,
            'detector_info': self.detector.get_model_info(),
        }
        return stats
    
    def reset(self):
        """Reset statistics."""
        self.frame_count = 0
        self.fps_times = []
        self.detection_times = []
        self.tracking_times = []
        self.tracker.reset()


class AerialDetector:
    """Helper class for color assignment (imported to avoid circular imports)."""
    
    @staticmethod
    def _id_to_color(track_id: int) -> Tuple[int, int, int]:
        """Generate consistent color for track ID."""
        # Use deterministic color based on ID
        np.random.seed(track_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        return color
