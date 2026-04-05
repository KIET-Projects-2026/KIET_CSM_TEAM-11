def save_video_with_anomalies(frames, anomaly_predictions, output_path, scale_factor=4, fps=10):
    """
    Save a video file with anomaly overlays for web display.
    Args:
        frames (np.ndarray): Preprocessed frames (N, H, W, C) normalized to [0, 1]
        anomaly_predictions (np.ndarray): Binary predictions (0 or 1) for each frame
        output_path (str): Path to save the output video
        scale_factor (int): Scale factor for output video size
        fps (int): Frames per second for output video
    """
    import cv2
    h, w = frames.shape[1:3]
    out_size = (w * scale_factor, h * scale_factor)
    # Use H.264/avc1 codec for better browser compatibility
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # fallback
    out = cv2.VideoWriter(output_path, fourcc, fps, out_size)
    for idx, frame in enumerate(frames):
        is_anomaly = anomaly_predictions[idx]
        display_frame = (frame * 255).astype('uint8')
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
        display_frame = cv2.resize(display_frame, out_size, interpolation=cv2.INTER_LINEAR)
        # Draw border
        color = (0, 0, 255) if is_anomaly else (0, 255, 0)
        thickness = 8 if is_anomaly else 3
        cv2.rectangle(display_frame, (0, 0), (out_size[0]-1, out_size[1]-1), color, thickness)
        # Add status text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 2
        status = 'ANOMALY' if is_anomaly else 'Normal'
        text_color = (0, 0, 255) if is_anomaly else (0, 255, 0)
        cv2.putText(display_frame, f'Status: {status}', (20, 40), font, font_scale, text_color, font_thickness)
        out.write(display_frame)
    out.release()
"""
Live Video Anomaly Detection Visualization Module

This module provides real-time video playback with anomaly detection overlays.
Uses OpenCV for visualization - no GUI frameworks required.

Features:
- Real-time video playback at configurable FPS
- Visual anomaly indicators (colored borders)
- Information overlays (frame number, error, status)
- Keyboard controls (q=quit, p=pause)
"""

import cv2
import numpy as np
import time


def play_video_with_anomalies(
    video_path,
    anomaly_predictions,
    errors,
    fps=30,
    scale_factor=4,
    flash_anomalies=True
):
    """
    Play video with real-time anomaly detection visualization.
    
    Args:
        video_path (str): Path to video file
        anomaly_predictions (np.ndarray): Binary predictions (0 or 1) for each frame
        errors (np.ndarray): Reconstruction errors per frame
        fps (int): Frames per second for playback (default: 30)
        scale_factor (int): Scale factor for display (default: 4, i.e., 128x128 -> 512x512)
        flash_anomalies (bool): Enable flashing effect for anomaly frames (default: True)
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print("\n" + "=" * 60)
    print("LIVE VIDEO ANOMALY DETECTION")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"Original FPS: {original_fps:.1f}")
    print(f"Playback FPS: {fps}")
    print(f"Anomalies detected: {np.sum(anomaly_predictions)} / {len(anomaly_predictions)}")
    print("\nControls:")
    print("  p / SPACE  → Pause/Resume")
    print("  q / ESC    → Quit")
    print("=" * 60 + "\n")
    
    # Playback state
    paused = False
    frame_idx = 0
    frame_delay = int(1000 / fps)  # Delay in milliseconds
    
    # For flashing effect
    flash_state = 0
    
    # Window name
    window_name = "Live Video Anomaly Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Main playback loop
    while True:
        if not paused:
            # Read next frame
            ret, frame = cap.read()
            
            if not ret or frame_idx >= len(anomaly_predictions):
                print("\n✓ Video playback completed")
                break
            
            # Get anomaly info for current frame
            is_anomaly = anomaly_predictions[frame_idx]
            error = errors[frame_idx]
            
            # Resize frame to 128x128 (preprocessing size)
            frame_resized = cv2.resize(frame, (128, 128))
            
            # Convert BGR to RGB and normalize
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Scale up for better visibility
            display_size = 128 * scale_factor
            display_frame = cv2.resize(frame_resized, (display_size, display_size), 
                                      interpolation=cv2.INTER_LINEAR)
            
            # Create annotated frame
            annotated_frame = display_frame.copy()
            
            # Add border based on anomaly status
            if is_anomaly:
                # Thick red border for anomalies
                border_color = (0, 0, 255)  # Red in BGR
                border_thickness = 8 * scale_factor // 4
                
                # Optional flashing effect
                if flash_anomalies:
                    flash_state = (flash_state + 1) % 6
                    if flash_state < 3:
                        border_color = (0, 0, 255)  # Red
                    else:
                        border_color = (0, 0, 128)  # Dark red
            else:
                # Thin green border for normal frames
                border_color = (0, 255, 0)  # Green in BGR
                border_thickness = 3 * scale_factor // 4
            
            # Draw border
            cv2.rectangle(annotated_frame, (0, 0), 
                         (display_size - 1, display_size - 1), 
                         border_color, border_thickness)
            
            # Add text overlays
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5 * (scale_factor / 4)
            font_thickness = max(1, scale_factor // 4)
            text_color = (255, 255, 255)  # White
            shadow_color = (0, 0, 0)  # Black shadow for readability
            
            y_offset = int(20 * scale_factor / 4)
            x_start = int(10 * scale_factor / 4)
            line_spacing = int(25 * scale_factor / 4)
            
            # Helper function to draw text with shadow
            def draw_text_with_shadow(img, text, pos, color):
                # Shadow
                cv2.putText(img, text, (pos[0] + 2, pos[1] + 2), 
                           font, font_scale, shadow_color, font_thickness)
                # Text
                cv2.putText(img, text, pos, 
                           font, font_scale, color, font_thickness)
            
            # Frame number
            draw_text_with_shadow(annotated_frame, 
                                f"Frame: {frame_idx + 1}/{len(anomaly_predictions)}", 
                                (x_start, y_offset), text_color)
            
            # Status
            y_offset += line_spacing
            if is_anomaly:
                status_text = "Status: ANOMALY"
                status_color = (0, 0, 255)  # Red
                draw_text_with_shadow(annotated_frame, status_text, 
                                    (x_start, y_offset), status_color)
                
                # Add warning emoji/text
                y_offset += line_spacing
                draw_text_with_shadow(annotated_frame, "!!! ANOMALY DETECTED !!!", 
                                    (x_start, y_offset), (0, 0, 255))
            else:
                status_text = "Status: Normal"
                status_color = (0, 255, 0)  # Green
                draw_text_with_shadow(annotated_frame, status_text, 
                                    (x_start, y_offset), status_color)
            
            # Reconstruction error
            y_offset += line_spacing
            draw_text_with_shadow(annotated_frame, f"Error: {error:.6f}", 
                                (x_start, y_offset), text_color)
            
            # Display frame
            cv2.imshow(window_name, annotated_frame)
            
            frame_idx += 1
        
        # Handle keyboard input
        key = cv2.waitKey(frame_delay if not paused else 100) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            print("\n✓ Video playback stopped by user")
            break
        elif key == ord('p') or key == 32:  # 'p' or SPACE
            paused = not paused
            status = "PAUSED" if paused else "PLAYING"
            print(f"→ {status} at frame {frame_idx}/{len(anomaly_predictions)}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("=" * 60)


def play_video_with_full_info(
    frames,
    anomaly_predictions,
    anomaly_scores,
    errors,
    threshold,
    fps=30,
    scale_factor=4,
    flash_anomalies=True
):
    """
    Play video with full anomaly detection information from preprocessed frames.
    
    This function is used when frames are already loaded and preprocessed.
    
    Args:
        frames (np.ndarray): Preprocessed frames (N, H, W, C) normalized to [0, 1]
        anomaly_predictions (np.ndarray): Binary predictions (0 or 1) for each frame
        anomaly_scores (np.ndarray): Normalized anomaly scores [0, 1]
        errors (np.ndarray): Reconstruction errors per frame
        threshold (float): Anomaly threshold value
        fps (int): Frames per second for playback (default: 30)
        scale_factor (int): Scale factor for display (default: 4)
        flash_anomalies (bool): Enable flashing effect for anomaly frames
    """
    print("\n" + "=" * 60)
    print("LIVE VIDEO ANOMALY DETECTION")
    print("=" * 60)
    print(f"Total frames: {len(frames)}")
    print(f"Playback FPS: {fps}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Anomalies detected: {np.sum(anomaly_predictions)} / {len(frames)}")
    print("\nControls:")
    print("  p / SPACE  → Pause/Resume")
    print("  q / ESC    → Quit")
    print("=" * 60 + "\n")
    
    # Playback state
    paused = False
    frame_idx = 0
    frame_delay = int(1000 / fps)
    flash_state = 0
    
    # Window setup
    window_name = "Live Video Anomaly Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Main playback loop
    while frame_idx < len(frames):
        if not paused:
            # Get current frame data
            frame = frames[frame_idx]
            is_anomaly = anomaly_predictions[frame_idx]
            anomaly_score = anomaly_scores[frame_idx]
            error = errors[frame_idx]
            
            # Convert from [0, 1] to [0, 255]
            display_frame = (frame * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
            
            # Scale up for visibility
            display_size = 128 * scale_factor
            display_frame = cv2.resize(display_frame, (display_size, display_size),
                                      interpolation=cv2.INTER_LINEAR)
            
            # Add border
            if is_anomaly:
                border_color = (0, 0, 255)  # Red
                border_thickness = 8 * scale_factor // 4
                
                if flash_anomalies:
                    flash_state = (flash_state + 1) % 6
                    border_color = (0, 0, 255) if flash_state < 3 else (0, 0, 128)
            else:
                border_color = (0, 255, 0)  # Green
                border_thickness = 3 * scale_factor // 4
            
            cv2.rectangle(display_frame, (0, 0),
                         (display_size - 1, display_size - 1),
                         border_color, border_thickness)
            
            # Add text overlays
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5 * (scale_factor / 4)
            font_thickness = max(1, scale_factor // 4)
            
            y_offset = int(20 * scale_factor / 4)
            x_start = int(10 * scale_factor / 4)
            line_spacing = int(25 * scale_factor / 4)
            
            def draw_text_with_shadow(img, text, pos, color):
                cv2.putText(img, text, (pos[0] + 2, pos[1] + 2),
                           font, font_scale, (0, 0, 0), font_thickness)
                cv2.putText(img, text, pos,
                           font, font_scale, color, font_thickness)
            
            # Frame number
            draw_text_with_shadow(display_frame,
                                f"Frame: {frame_idx + 1}/{len(frames)}",
                                (x_start, y_offset), (255, 255, 255))
            
            # Status
            y_offset += line_spacing
            if is_anomaly:
                draw_text_with_shadow(display_frame, "Status: ANOMALY",
                                    (x_start, y_offset), (0, 0, 255))
                y_offset += line_spacing
                draw_text_with_shadow(display_frame, "!!! ANOMALY DETECTED !!!",
                                    (x_start, y_offset), (0, 0, 255))
            else:
                draw_text_with_shadow(display_frame, "Status: Normal",
                                    (x_start, y_offset), (0, 255, 0))
            
            # Error and score
            y_offset += line_spacing
            draw_text_with_shadow(display_frame, f"Error: {error:.6f}",
                                (x_start, y_offset), (255, 255, 255))
            
            y_offset += line_spacing
            draw_text_with_shadow(display_frame, f"Score: {anomaly_score:.4f}",
                                (x_start, y_offset), (255, 255, 255))
            
            # Display
            cv2.imshow(window_name, display_frame)
            frame_idx += 1
        
        # Keyboard handling
        key = cv2.waitKey(frame_delay if not paused else 100) & 0xFF
        
        if key == ord('q') or key == 27:
            print(f"\n✓ Stopped at frame {frame_idx}/{len(frames)}")
            break
        elif key == ord('p') or key == 32:
            paused = not paused
            print(f"→ {'PAUSED' if paused else 'PLAYING'} at frame {frame_idx}/{len(frames)}")
    
    if frame_idx >= len(frames):
        print("\n✓ Video playback completed")
    
    cv2.destroyAllWindows()
    print("=" * 60)
