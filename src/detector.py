"""
Lightweight detector optimized for small-scale objects in aerial imagery.
Uses YOLOv8 Nano for efficiency while maintaining reasonable accuracy.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional
import torch

class AerialDetector:
    """
    Detector optimized for person detection in drone footage.
    Uses YOLOv8n (Nano) - <50MB model for edge deployment.
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.45):
        """
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # Person class ID in COCO dataset
        self.person_class_id = 0
        
    def detect(self, frame: np.ndarray, target_size: Optional[int] = None) -> np.ndarray:
        """
        Detect persons in frame.
        
        Args:
            frame: Input image
            target_size: Resize frame for faster inference (e.g., 640)
            
        Returns:
            Array of detections [N, 6] where columns are:
            [x1, y1, x2, y2, confidence, class_id]
        """
        if target_size is not None:
            h, w = frame.shape[:2]
            scale = target_size / max(h, w)
            if scale < 1:
                new_w, new_h = int(w * scale), int(h * scale)
                frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                scale_factor = h / new_h
            else:
                frame_resized = frame
                scale_factor = 1.0
        else:
            frame_resized = frame
            scale_factor = 1.0
        
        # Run inference
        results = self.model(frame_resized, conf=self.conf_threshold, classes=[self.person_class_id], verbose=False)
        
        # Extract detections
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls_id = box.cls[0]
                
                # Scale back to original frame size
                x1, y1, x2, y2 = x1 * scale_factor, y1 * scale_factor, x2 * scale_factor, y2 * scale_factor
                detections.append([x1, y1, x2, y2, conf, cls_id])
        
        return np.array(detections) if detections else np.empty((0, 6))
    
    def detect_with_nms(self, frame: np.ndarray, nms_threshold: float = 0.45,
                        target_size: Optional[int] = None) -> np.ndarray:
        """
        Detect with custom NMS to handle small objects better.
        
        Args:
            frame: Input image
            nms_threshold: NMS IoU threshold
            target_size: Resize for inference
            
        Returns:
            Array of detections after NMS
        """
        detections = self.detect(frame, target_size)
        
        if len(detections) == 0:
            return detections
        
        # Apply NMS
        keep_indices = self._soft_nms(detections, nms_threshold)
        return detections[keep_indices]
    
    @staticmethod
    def _soft_nms(detections: np.ndarray, threshold: float = 0.45) -> np.ndarray:
        """Soft-NMS to better handle dense small objects."""
        if len(detections) == 0:
            return np.array([], dtype=int)
        
        boxes = detections[:, :4]
        scores = detections[:, 4]
        
        # Sort by score
        sorted_indices = np.argsort(-scores)
        keep = []
        suppression = np.zeros(len(detections))
        
        for i in sorted_indices:
            if suppression[i] == 0:
                keep.append(i)
                # Calculate IoU with remaining boxes
                for j in sorted_indices:
                    if i != j and suppression[j] == 0:
                        iou = AerialDetector._iou(boxes[i], boxes[j])
                        if iou > threshold:
                            # Soft-NMS: reduce confidence
                            suppression[j] = max(suppression[j], iou)
        
        return np.array(keep)
    
    @staticmethod
    def _iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes."""
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
    
    def get_model_info(self) -> dict:
        """Get model size and parameter info."""
        total_params = sum(p.numel() for p in self.model.model.parameters())
        return {
            "model_name": "YOLOv8n (Nano)",
            "total_parameters": total_params,
            "device": self.device
        }
