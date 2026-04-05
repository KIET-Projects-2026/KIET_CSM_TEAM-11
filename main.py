"""
Video Feature Extraction Pipeline Runner

This script orchestrates the complete feature extraction pipeline:
1. Loads video frames
2. Builds CNN autoencoder
3. Extracts deep features
4. Prints pipeline statistics

Supports both video file mode and live webcam mode.

Usage:
    # Video file mode (default)
    python main.py --mode video --video sample_video.mp4
    
    # Webcam mode
    python main.py --mode webcam --camera 0
"""

import numpy as np
import cv2
import argparse
from feature_extractor import (
    load_video_frames,
    build_autoencoder,
    extract_video_features
)
from anomaly_detector import (
    compute_reconstruction_error,
    detect_anomalies,
    compute_anomaly_scores
)
from video_visualizer import play_video_with_full_info
from webcam_detector import run_webcam_detection
from collections import deque
import time


def run_feature_extraction_pipeline(video_path, frame_skip=5, latent_dim=256):
    """
    Run the complete video feature extraction pipeline.
    
    Args:
        video_path (str): Path to video file
        frame_skip (int): Extract every Nth frame
        latent_dim (int): Dimension of latent feature vector
    
    Returns:
        tuple: (frames, features, encoder, autoencoder)
    """
    print("=" * 60)
    print("VIDEO FEATURE EXTRACTION PIPELINE")
    print("=" * 60)
    
    # Step 1: Load video frames
    print(f"\n[1/3] Loading video frames from: {video_path}")
    print(f"       Frame skip: {frame_skip}")
    frames = load_video_frames(video_path, frame_skip=frame_skip)
    print(f"       ✓ Loaded {len(frames)} frames")
    print(f"       ✓ Frame shape: {frames.shape}")
    
    # Step 2: Build autoencoder
    print(f"\n[2/3] Building CNN Autoencoder")
    print(f"       Input shape: (128, 128, 3)")
    print(f"       Latent dimension: {latent_dim}")
    autoencoder, encoder = build_autoencoder(
        input_shape=(128, 128, 3),
        latent_dim=latent_dim
    )
    print(f"       ✓ Autoencoder built successfully")
    print(f"       ✓ Total parameters: {autoencoder.count_params():,}")
    
    # Step 3: Extract features
    print(f"\n[3/3] Extracting deep features")
    features = extract_video_features(encoder, frames)
    print(f"       ✓ Feature extraction complete")
    print(f"       ✓ Feature shape: {features.shape}")
    print(f"       ✓ Feature vector dimension: {features.shape[1]}")
    
    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Frames processed:          {len(frames)}")
    print(f"Features extracted:        {len(features)}")
    print(f"Feature vector shape:      {features.shape}")
    print(f"Memory footprint:          {features.nbytes / (1024**2):.2f} MB")
    print("=" * 60)
    
    return frames, features, encoder, autoencoder


def visualize_anomaly_detection(frames, anomaly_predictions, anomaly_scores, errors, threshold):
    """
    Display a window showing frames with anomaly detection visualization.
    
    Args:
        frames: Original frames (N, H, W, C) normalized to [0, 1]
        anomaly_predictions: Binary predictions (0 or 1) for each frame
        anomaly_scores: Normalized anomaly scores [0, 1]
        errors: Reconstruction errors
        threshold: Anomaly threshold value
    """
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION VISUALIZATION")
    print("=" * 60)
    print(f"Threshold: {threshold:.6f}")
    print(f"Anomalies detected: {np.sum(anomaly_predictions)} / {len(frames)}")
    print("Press SPACE to play/pause, ESC to exit, or click to go frame by frame")
    print("=" * 60 + "\n")
    
    paused = True
    frame_idx = 0
    
    while True:
        if frame_idx >= len(frames):
            break
        
        # Get current frame
        frame = frames[frame_idx]
        is_anomaly = anomaly_predictions[frame_idx]
        anomaly_score = anomaly_scores[frame_idx]
        error = errors[frame_idx]
        
        # Convert frame from [0, 1] to [0, 255] for display
        display_frame = (frame * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
        
        # Create a copy for annotations
        annotated_frame = display_frame.copy()
        
        # Add border color based on anomaly status
        border_color = (0, 0, 255) if is_anomaly else (0, 255, 0)  # Red for anomaly, Green for normal
        border_thickness = 5
        cv2.rectangle(annotated_frame, (0, 0), (128, 128), border_color, border_thickness)
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        text_color = (255, 255, 255)  # White text
        
        # Frame number
        cv2.putText(annotated_frame, f"Frame: {frame_idx + 1}/{len(frames)}", 
                   (5, 15), font, font_scale, text_color, font_thickness)
        
        # Anomaly status
        status_text = "ANOMALY" if is_anomaly else "NORMAL"
        status_color = (0, 0, 255) if is_anomaly else (0, 255, 0)
        cv2.putText(annotated_frame, f"Status: {status_text}", 
                   (5, 35), font, font_scale, status_color, font_thickness)
        
        # Anomaly score
        cv2.putText(annotated_frame, f"Score: {anomaly_score:.4f}", 
                   (5, 55), font, font_scale, text_color, font_thickness)
        
        # Reconstruction error
        cv2.putText(annotated_frame, f"Error: {error:.6f}", 
                   (5, 75), font, font_scale, text_color, font_thickness)
        
        # Threshold
        cv2.putText(annotated_frame, f"Threshold: {threshold:.6f}", 
                   (5, 95), font, font_scale, text_color, font_thickness)
        
        # Display the frame
        cv2.imshow("Anomaly Detection Visualization", annotated_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(100 if not paused else 0) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE - play/pause
            paused = not paused
        elif key == 82 or key == 83:  # Arrow keys for frame navigation
            # Right arrow
            if key == 83 and frame_idx < len(frames) - 1:
                frame_idx += 1
            # Left arrow
            elif key == 82 and frame_idx > 0:
                frame_idx -= 1
        elif not paused:
            frame_idx += 1
    
    cv2.destroyAllWindows()


def main():
    """
    Main entry point for the complete anomaly detection pipeline.
    Supports both video file and webcam modes.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Video Anomaly Detection - Supports video files and live webcam"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["video", "webcam"],
        default="video",
        help="Detection mode: 'video' for file, 'webcam' for live feed (default: video)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="sample_video.mp4",
        help="Path to video file (for video mode, default: sample_video.mp4)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (for webcam mode, default: 0)"
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=256,
        help="Latent dimension for autoencoder (default: 256)"
    )
    
    args = parser.parse_args()
    
    try:
        # Build autoencoder (needed for both modes)
        print("=" * 60)
        print("VIDEO ANOMALY DETECTION SYSTEM")
        print("=" * 60)
        print(f"Mode: {args.mode.upper()}")
        print("=" * 60)
        
        print(f"\n[1/2] Building CNN Autoencoder")
        print(f"       Input shape: (128, 128, 3)")
        print(f"       Latent dimension: {args.latent_dim}")
        autoencoder, encoder = build_autoencoder(
            input_shape=(128, 128, 3),
            latent_dim=args.latent_dim
        )
        print(f"       ✓ Autoencoder built successfully")
        print(f"       ✓ Total parameters: {autoencoder.count_params():,}")
        
        if args.mode == "webcam":
            # Webcam mode
            print(f"\n[2/2] Launching webcam detection")
            print(f"       Camera ID: {args.camera}")
            print("\n🎥 Starting live webcam detection...")
            print("   (Calibration will start automatically)\n")
            
            run_webcam_detection(
                autoencoder=autoencoder,
                encoder=encoder,
                camera_id=args.camera
            )
            
            print("\n✓ Webcam detection completed!")
            
        else:
            # Video file mode
            video_path = args.video
            
            print(f"\n[2/2] Processing video file")
            print(f"       Video: {video_path}")
            
            # Step 1: Feature extraction pipeline
            frames, features, encoder, autoencoder = run_feature_extraction_pipeline(
                video_path=video_path,
                frame_skip=5,
                latent_dim=args.latent_dim
            )
            
            print("\n✓ Feature extraction completed successfully!")
            
            # Step 2: Anomaly detection
            print("\n" + "=" * 60)
            print("ANOMALY DETECTION")
            print("=" * 60)
            
            # Compute reconstruction errors
            print("\n[1/3] Computing reconstruction errors...")
            errors = compute_reconstruction_error(autoencoder, frames)
            print(f"       ✓ Errors computed: min={errors.min():.6f}, max={errors.max():.6f}")
            
            # Detect anomalies
            print("\n[2/3] Detecting anomalies...")
            anomaly_predictions, threshold = detect_anomalies(errors, percentile=95)
            print(f"       ✓ Threshold: {threshold:.6f}")
            print(f"       ✓ Anomalies detected: {np.sum(anomaly_predictions)} / {len(frames)}")
            
            # Compute normalized anomaly scores
            print("\n[3/3] Computing anomaly scores...")
            anomaly_scores = compute_anomaly_scores(errors)
            print(f"       ✓ Scores computed: min={anomaly_scores.min():.4f}, max={anomaly_scores.max():.4f}")
            
            print("=" * 60)
            
            # Step 3: Launch live video visualization
            print("\n🎬 Launching live video visualization...")
            print("   (Window will open in a moment)")
            
            play_video_with_full_info(
                frames=frames,
                anomaly_predictions=anomaly_predictions,
                anomaly_scores=anomaly_scores,
                errors=errors,
                threshold=threshold,
                fps=10,  # Slower playback for better observation
                scale_factor=4,  # 128x128 -> 512x512
                flash_anomalies=True
            )
            
            print("\n✓ Pipeline completed successfully!")
        
    except FileNotFoundError:
        print(f"\n✗ Error: Video file not found: {args.video}")
        print("   Please check the file path and try again")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
