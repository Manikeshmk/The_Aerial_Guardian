"""
ByteTrack Multi-Object Tracker with drone ego-motion compensation.
Handles ID switching caused by camera motion and occlusions.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import cv2

@dataclass
class TrackState:
    """State representation for a tracked object."""
    id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    age: int  # frames since track created
    hits: int  # detections in last n frames
    hit_streak: int  # consecutive detections
    last_detection_frame: int
    kalman_filter: Optional[object] = None
    trajectory: List[Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.trajectory is None:
            self.trajectory = []


class DroneByteTrac:
    """
    ByteTrack algorithm optimized for drone footage.
    Features:
    - Handles dense small objects
    - Ego-motion compensation
    - Reduced ID switching via trajectory smoothing
    """
    
    def __init__(self, 
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.1,
                 track_activation_threshold: float = 0.6):
        """
        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum detections to activate track
            iou_threshold: IoU threshold for matching
            track_activation_threshold: Confidence threshold for track activation
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.track_activation_threshold = track_activation_threshold
        
        self.tracks: List[TrackState] = []
        self.next_id = 1
        self.frame_count = 0
        self.prev_frame = None
        
        # Ego-motion estimation
        self.optical_flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.ego_motion = np.eye(3)  # Homography for ego-motion
        
    def update(self, detections: np.ndarray, frame: np.ndarray) -> List[Dict]:
        """
        Update tracks with new detections.
        
        Args:
            detections: Array [N, 6] with [x1, y1, x2, y2, conf, class_id]
            frame: Current frame for optical flow
            
        Returns:
            List of updated tracks with IDs
        """
        self.frame_count += 1
        
        # Estimate ego-motion
        ego_motion_delta = self._estimate_ego_motion(frame)
        
        # Separate high and low confidence detections
        high_conf = detections[detections[:, 4] >= self.track_activation_threshold]
        low_conf = detections[detections[:, 4] < self.track_activation_threshold]
        
        # Match high confidence detections to tracks
        if len(self.tracks) > 0:
            matched_pairs, unmatched_tracks, unmatched_detections = self._match_detections(
                high_conf[:, :4], self.tracks, ego_motion_delta
            )
        else:
            matched_pairs = []
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_detections = list(range(len(high_conf)))
        
        # Update matched tracks
        for track_idx, det_idx in matched_pairs:
            track = self.tracks[track_idx]
            det = high_conf[det_idx]
            track.bbox = det[:4]
            track.confidence = det[4]
            track.hits += 1
            track.hit_streak += 1
            track.last_detection_frame = self.frame_count
            # Update trajectory
            center = self._get_center(track.bbox)
            track.trajectory.append(center)
            if len(track.trajectory) > 50:  # Keep last 50 positions
                track.trajectory.pop(0)
        
        # Create new tracks from unmatched detections
        for det_idx in unmatched_detections:
            det = high_conf[det_idx]
            track = TrackState(
                id=self.next_id,
                bbox=det[:4],
                confidence=det[4],
                age=1,
                hits=1,
                hit_streak=1,
                last_detection_frame=self.frame_count
            )
            track.trajectory = [self._get_center(track.bbox)]
            self.tracks.append(track)
            self.next_id += 1
        
        # Update unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].age += 1
            self.tracks[track_idx].hit_streak = 0
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks 
                      if t.age < self.max_age and 
                      (self.frame_count - t.last_detection_frame) < self.max_age]
        
        # Prepare output
        active_tracks = []
        for track in self.tracks:
            if track.hits >= self.min_hits or self.frame_count < self.min_hits:
                x1, y1, x2, y2 = track.bbox
                active_tracks.append({
                    'track_id': track.id,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': track.confidence,
                    'age': track.age,
                    'trajectory': track.trajectory[-10:]  # Last 10 positions for visualization
                })
        
        self.prev_frame = frame.copy()
        return active_tracks
    
    def _estimate_ego_motion(self, frame: np.ndarray) -> np.ndarray:
        """Estimate camera motion between frames using optical flow."""
        if self.prev_frame is None:
            return np.zeros((2,))
        
        # Reduce frame size for faster computation
        h, w = frame.shape[:2]
        scale = 0.5
        frame_small = cv2.resize(frame, (int(w*scale), int(h*scale)))
        prev_small = cv2.resize(self.prev_frame, (int(w*scale), int(h*scale)))
        
        # Convert to grayscale
        frame_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
        
        try:
            # Compute optical flow
            flow = self.optical_flow.calc(prev_gray, frame_gray, None)
            # Get median flow as ego-motion estimate
            flow_magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
            valid_flow = flow[flow_magnitude < 10]  # Remove outliers
            if len(valid_flow) > 0:
                ego_motion = np.median(valid_flow, axis=0)
            else:
                ego_motion = np.array([0., 0.])
        except:
            ego_motion = np.array([0., 0.])
        
        return ego_motion
    
    def _match_detections(self, detections: np.ndarray, tracks: List[TrackState],
                         ego_motion: np.ndarray) -> Tuple[List, List, List]:
        """
        Match detections to tracks using IoU with ego-motion compensation.
        Uses Hungarian algorithm for optimal assignment.
        """
        if len(tracks) == 0:
            return [], [], list(range(len(detections)))
        
        # Compute IoU matrix with ego-motion compensation
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for t_idx, track in enumerate(tracks):
            # Compensate track position for ego-motion
            compensated_bbox = self._compensate_bbox(track.bbox, ego_motion)
            for d_idx, detection in enumerate(detections):
                iou_matrix[t_idx, d_idx] = self._iou(compensated_bbox, detection[:4])
        
        # Use Hungarian algorithm for assignment
        track_indices, det_indices = linear_sum_assignment(-iou_matrix)
        
        # Filter matches below threshold
        matched_pairs = []
        for t_idx, d_idx in zip(track_indices, det_indices):
            if iou_matrix[t_idx, d_idx] > self.iou_threshold:
                matched_pairs.append((t_idx, d_idx))
        
        matched_track_indices = set([p[0] for p in matched_pairs])
        matched_det_indices = set([p[1] for p in matched_pairs])
        
        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_track_indices]
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_det_indices]
        
        return matched_pairs, unmatched_tracks, unmatched_detections
    
    @staticmethod
    def _compensate_bbox(bbox: np.ndarray, ego_motion: np.ndarray) -> np.ndarray:
        """Adjust bounding box for estimated ego-motion."""
        x1, y1, x2, y2 = bbox
        dx, dy = ego_motion
        return np.array([x1 + dx, y1 + dy, x2 + dx, y2 + dy])
    
    @staticmethod
    def _iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    @staticmethod
    def _get_center(bbox: np.ndarray) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.frame_count = 0
        self.prev_frame = None
        self.next_id = 1
