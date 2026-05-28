#!/usr/bin/env python
"""
Main pipeline for Aerial Person Detection and Tracking.
Processes drone footage to detect and track persons with ID maintenance.
"""

import cv2
import numpy as np
import argparse
import time
import os
from pathlib import Path
import logging

from src.detector import AerialDetector
from src.tracker import DroneByteTrac
from src.pipeline import DroneVideoProcessor, AerialPreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_video(video_path: str, output_path: str, model_path: str = "yolov8n.pt",
                  conf_threshold: float = 0.45, max_age: int = 30, 
                  skip_frames: int = 0, preview: bool = False):
    """
    Process video for person detection and tracking.
    
    Args:
        video_path: Path to input video
        output_path: Path to output video
        model_path: Path to YOLO model weights
        conf_threshold: Detection confidence threshold
        max_age: Max frames to keep track without detection
        skip_frames: Process every Nth frame
        preview: Show live preview
    """
    
    # Initialize components
    logger.info("Initializing detector...")
    detector = AerialDetector(model_path=model_path, conf_threshold=conf_threshold)
    
    logger.info("Initializing tracker...")
    tracker = DroneByteTrac(max_age=max_age, min_hits=3, iou_threshold=0.1)
    
    logger.info("Initializing preprocessor...")
    preprocessor = AerialPreprocessor(enable_clahe=True, enable_sharpening=True)
    
    # Create pipeline
    pipeline = DroneVideoProcessor(detector, tracker, preprocessor)
    
    # Open video
    logger.info(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup output video
    output_fps = fps / max(1, skip_frames + 1)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    
    frame_count = 0
    processed_frames = 0
    processing_times = []
    
    logger.info("Processing video...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames if specified
            if skip_frames > 0 and (frame_count - 1) % (skip_frames + 1) != 0:
                continue
            
            # Process frame
            start_time = time.time()
            frame_vis, tracks = pipeline.process_frame(frame, apply_preprocessing=True)
            process_time = time.time() - start_time
            processing_times.append(process_time)
            processed_frames += 1
            
            # Write output frame
            out.write(frame_vis)
            
            # Print progress
            if processed_frames % 30 == 0:
                avg_time = np.mean(processing_times[-30:])
                current_fps = 1.0 / avg_time if avg_time > 0 else 0
                logger.info(f"Frame {frame_count}/{total_frames} | "
                          f"Detections: {len(tracks)} | "
                          f"FPS: {current_fps:.2f}")
            
            # Preview if requested
            if preview:
                display_frame = cv2.resize(frame_vis, (1024, 768))
                cv2.imshow('Processing', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    finally:
        cap.release()
        out.release()
        if preview:
            cv2.destroyAllWindows()
    
    # Print statistics
    logger.info("\n" + "="*60)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*60)
    
    stats = pipeline.get_statistics()
    logger.info(f"Total frames processed: {processed_frames}")
    logger.info(f"Average FPS: {stats['avg_fps']:.2f}")
    logger.info(f"Average detection time: {stats['avg_detection_time']*1000:.2f}ms")
    logger.info(f"Average tracking time: {stats['avg_tracking_time']*1000:.2f}ms")
    logger.info(f"Detector: {stats['detector_info']['model_name']}")
    logger.info(f"Total parameters: {stats['detector_info']['total_parameters']:,}")
    logger.info(f"Device: {stats['detector_info']['device']}")
    logger.info(f"Output video: {output_path}")
    logger.info("="*60 + "\n")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Aerial Person Detection and Tracking Pipeline')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, default='output/tracked_video.mp4', help='Output video path')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--conf', type=float, default=0.45, help='Detection confidence threshold')
    parser.add_argument('--max-age', type=int, default=30, help='Max frames to keep track')
    parser.add_argument('--skip-frames', type=int, default=0, help='Process every Nth frame')
    parser.add_argument('--preview', action='store_true', help='Show live preview')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process video
    stats = process_video(
        args.video,
        args.output,
        model_path=args.model,
        conf_threshold=args.conf,
        max_age=args.max_age,
        skip_frames=args.skip_frames,
        preview=args.preview
    )


if __name__ == '__main__':
    main()
