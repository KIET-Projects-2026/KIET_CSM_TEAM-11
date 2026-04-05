"""
Live Webcam Anomaly Detection Module

This module provides real-time anomaly detection from webcam feed.
Features:
- Real-time frame capture and preprocessing
- Frame buffer management for temporal context
- Live anomaly detection with adaptive thresholding
- Real-time visualization with overlays
- Keyboard controls (ESC=quit, SPACE=pause, 'c'=calibrate)
"""

import cv2
import numpy as np
from collections import deque
import time


class WebcamAnomalyDetector:
    """
    Real-time webcam anomaly detector using autoencoder-based reconstruction error.
    """
    
    def __init__(self, autoencoder, encoder, buffer_size=30, calibration_frames=50):
        """
        Initialize webcam anomaly detector.
        
        Args:
            autoencoder: Trained autoencoder model for reconstruction
            encoder: Encoder model for feature extraction
            buffer_size (int): Number of frames to keep in rolling buffer
            calibration_frames (int): Number of frames for initial calibration
        """
        self.autoencoder = autoencoder
        self.encoder = encoder
        self.buffer_size = buffer_size
        self.calibration_frames = calibration_frames
        
        # Frame buffer for temporal analysis
        self.frame_buffer = deque(maxlen=buffer_size)
        
        # Error statistics for adaptive thresholding
        self.error_buffer = deque(maxlen=buffer_size)
        self.mean_error = 0.0
        self.std_error = 0.0
        self.threshold = 0.0
        
        # Calibration state
        self.is_calibrated = False
        self.calibration_errors = []
        
        # Display settings
        self.display_size = 512
        self.flash_state = 0
        
    def calibrate(self, cap, num_frames=None):
        """
        Calibrate detector by collecting baseline statistics from normal frames.
        
        Args:
            cap: OpenCV VideoCapture object
            num_frames (int): Number of frames for calibration (default: self.calibration_frames)
        """
        if num_frames is None:
            num_frames = self.calibration_frames
            
        print(f"\n{'='*60}")
        print("CALIBRATING DETECTOR")
        print(f"{'='*60}")
        print(f"Please ensure normal activity is in view...")
        print(f"Collecting {num_frames} frames for baseline...\n")
        
        self.calibration_errors = []
        
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to capture frame during calibration")
                continue
                
            # Preprocess frame
            processed_frame = self._preprocess_frame(frame)
            
            # Compute reconstruction error
            error = self._compute_error(processed_frame)
            self.calibration_errors.append(error)
            
            # Show progress
            if (i + 1) % 10 == 0:
                print(f"  Collected {i + 1}/{num_frames} frames...")
        
        # Compute baseline statistics
        if len(self.calibration_errors) > 0:
            self.mean_error = np.mean(self.calibration_errors)
            self.std_error = np.std(self.calibration_errors)
            # Set threshold at mean + 3 standard deviations (99.7% confidence)
            self.threshold = self.mean_error + 3 * self.std_error
            self.is_calibrated = True
            
            print(f"\n✓ Calibration complete!")
            print(f"  Mean error: {self.mean_error:.6f}")
            print(f"  Std error: {self.std_error:.6f}")
            print(f"  Threshold: {self.threshold:.6f}")
            print(f"{'='*60}\n")
        else:
            print("✗ Calibration failed - no frames collected")
    
    def _preprocess_frame(self, frame):
        """
        Preprocess webcam frame to match model input requirements.
        
        Args:
            frame: Raw BGR frame from webcam
            
        Returns:
            np.ndarray: Preprocessed frame (128, 128, 3) normalized to [0, 1]
        """
        # Resize to 128x128
        resized = cv2.resize(frame, (128, 128))
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb_frame.astype(np.float32) / 255.0
        
        return normalized
    
    def _compute_error(self, frame):
        """
        Compute reconstruction error for a single frame.
        
        Args:
            frame: Preprocessed frame (128, 128, 3)
            
        Returns:
            float: Reconstruction error
        """
        # Add batch dimension
        frame_batch = np.expand_dims(frame, axis=0)
        
        # Reconstruct frame
        reconstructed = self.autoencoder.predict(frame_batch, verbose=0)
        
        # Compute MSE
        error = np.mean((frame_batch - reconstructed) ** 2)
        
        return error
    
    def detect_anomaly(self, error):
        """
        Detect if current error indicates an anomaly.
        
        Args:
            error (float): Reconstruction error
            
        Returns:
            tuple: (is_anomaly, anomaly_score)
        """
        if not self.is_calibrated:
            return False, 0.0
        
        # Determine if anomaly
        is_anomaly = error > self.threshold
        
        # Compute normalized score
        if self.std_error > 0:
            anomaly_score = (error - self.mean_error) / (3 * self.std_error)
            anomaly_score = np.clip(anomaly_score, 0.0, 1.0)
        else:
            anomaly_score = 0.0
        
        return is_anomaly, anomaly_score
    
    def update_statistics(self, error):
        """
        Update rolling statistics with new error value.
        
        Args:
            error (float): New reconstruction error
        """
        self.error_buffer.append(error)
        
        if len(self.error_buffer) > 10:  # Only update after collecting some samples
            self.mean_error = np.mean(self.error_buffer)
            self.std_error = np.std(self.error_buffer)
            # Update threshold with rolling window
            self.threshold = self.mean_error + 3 * self.std_error
    
    def generate_frames(self, camera_id=0, auto_calibrate=True):
        """
        Generator function that yields JPEG frames for MJPEG streaming.
        
        Args:
            camera_id (int): Camera device ID
            auto_calibrate (bool): Automatically calibrate on startup
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Cannot open webcam {camera_id}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        frame_count = 0
        
        # Auto-calibration logic inline
        if auto_calibrate:
            # We will calibrate "on the fly" for the first N frames
            # But to keep the stream going, we'll just collect data
            pass

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calibration phase
                if not self.is_calibrated:
                    if len(self.calibration_errors) < self.calibration_frames:
                        processed_frame = self._preprocess_frame(frame)
                        error = self._compute_error(processed_frame)
                        self.calibration_errors.append(error)
                        
                        # Show calibration progress
                        display_frame = cv2.resize(frame, (self.display_size, self.display_size))
                        cv2.putText(display_frame, f"CALIBRATING: {len(self.calibration_errors)}/{self.calibration_frames}", 
                                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        
                        # Encode
                        ret, buffer = cv2.imencode('.jpg', display_frame)
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        continue
                    else:
                        # Finish calibration
                        self.mean_error = np.mean(self.calibration_errors)
                        self.std_error = np.std(self.calibration_errors)
                        self.threshold = self.mean_error + 3 * self.std_error
                        self.is_calibrated = True
                        print("Calibration complete via stream")

                # Normal detection phase
                processed_frame = self._preprocess_frame(frame)
                error = self._compute_error(processed_frame)
                is_anomaly, anomaly_score = self.detect_anomaly(error)
                self.update_statistics(error)
                self.frame_buffer.append(processed_frame)
                
                display_frame = self._create_display_frame(
                    frame, processed_frame, error, is_anomaly, 
                    anomaly_score, frame_count
                )
                frame_count += 1
                
                # Encode to JPEG
                ret, buffer = cv2.imencode('.jpg', display_frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                       
        finally:
            cap.release()
    
    def _create_display_frame(self, original_frame, processed_frame, error, 
                             is_anomaly, anomaly_score, frame_count):
        """
        Create annotated display frame with all overlays.
        
        Args:
            original_frame: Original BGR frame from webcam
            processed_frame: Preprocessed frame
            error: Reconstruction error
            is_anomaly: Boolean anomaly flag
            anomaly_score: Normalized anomaly score
            frame_count: Current frame number
            
        Returns:
            np.ndarray: Annotated BGR frame for display
        """
        # Resize original frame for display
        display_frame = cv2.resize(original_frame, (self.display_size, self.display_size))
        
        # Add border based on anomaly status
        if is_anomaly and self.is_calibrated:
            # Red border for anomalies with flashing effect
            self.flash_state = (self.flash_state + 1) % 6
            border_color = (0, 0, 255) if self.flash_state < 3 else (0, 0, 128)
            border_thickness = 8
        elif self.is_calibrated:
            # Green border for normal frames
            border_color = (0, 255, 0)
            border_thickness = 3
        else:
            # Yellow border when not calibrated
            border_color = (0, 255, 255)
            border_thickness = 3
        
        cv2.rectangle(display_frame, (0, 0), 
                     (self.display_size - 1, self.display_size - 1),
                     border_color, border_thickness)
        
        # Add text overlays
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        y_offset = 30
        x_start = 15
        line_spacing = 35
        
        def draw_text_with_shadow(img, text, pos, color):
            # Shadow for readability
            cv2.putText(img, text, (pos[0] + 2, pos[1] + 2),
                       font, font_scale, (0, 0, 0), font_thickness)
            cv2.putText(img, text, pos,
                       font, font_scale, color, font_thickness)
        
        # Frame counter
        draw_text_with_shadow(display_frame, f"Frame: {frame_count}",
                            (x_start, y_offset), (255, 255, 255))
        
        # Calibration status
        y_offset += line_spacing
        if not self.is_calibrated:
            draw_text_with_shadow(display_frame, "Status: NOT CALIBRATED",
                                (x_start, y_offset), (0, 255, 255))
            y_offset += line_spacing
            draw_text_with_shadow(display_frame, "Press 'c' to calibrate",
                                (x_start, y_offset), (0, 255, 255))
        else:
            # Anomaly status
            if is_anomaly:
                draw_text_with_shadow(display_frame, "Status: ANOMALY DETECTED",
                                    (x_start, y_offset), (0, 0, 255))
                y_offset += line_spacing
                draw_text_with_shadow(display_frame, "!!! WARNING !!!",
                                    (x_start, y_offset), (0, 0, 255))
            else:
                draw_text_with_shadow(display_frame, "Status: Normal",
                                    (x_start, y_offset), (0, 255, 0))
            
            # Error value
            y_offset += line_spacing
            draw_text_with_shadow(display_frame, f"Error: {error:.6f}",
                                (x_start, y_offset), (255, 255, 255))
            
            # Threshold
            y_offset += line_spacing
            draw_text_with_shadow(display_frame, f"Threshold: {self.threshold:.6f}",
                                (x_start, y_offset), (255, 255, 255))
            
            # Anomaly score
            y_offset += line_spacing
            score_color = (0, 0, 255) if anomaly_score > 0.5 else (255, 255, 255)
            draw_text_with_shadow(display_frame, f"Score: {anomaly_score:.2%}",
                                (x_start, y_offset), score_color)
        
        return display_frame


class OpenCVAnomalyDetector:
    """
    Real-time anomaly detection using OpenCV Background Subtraction (MOG2).
    No deep learning model required.
    """
    
    def __init__(self, history=500, varThreshold=16, detectShadows=True):
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=detectShadows)
        self.display_size = 512
        self.min_contour_area = 500  # Minimum area to consider as anomaly
        
    def generate_frames(self, camera_id=0):
        print(f"[DEBUG] Attempting to open camera {camera_id}...")
        # Try using DirectShow on Windows (often fixes black screen/no access issues)
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print(f"[ERROR] Failed to open camera {camera_id} with CAP_DSHOW. Trying default backend...")
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                print(f"[ERROR] Cannot open webcam {camera_id}")
                return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"[DEBUG] Camera {camera_id} opened successfully.")
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to read frame from camera.")
                    break
                
                if frame_count % 30 == 0:
                    print(f"[DEBUG] Processing frame {frame_count}")

                # Resize for consistent processing
                display_frame = cv2.resize(frame, (self.display_size, self.display_size))
                
                # Apply background subtraction
                fgMask = self.backSub.apply(display_frame)
                
                # Threshold the mask to remove shadows (shadows are gray in MOG2)
                _, fgMask = cv2.threshold(fgMask, 250, 255, cv2.THRESH_BINARY)
                
                # Find contours of moving objects
                contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                is_anomaly = False
                anomaly_score = 0.0
                max_area = 0
                
                # Draw bounding boxes around anomalies
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > self.min_contour_area:
                        is_anomaly = True
                        max_area = max(max_area, area)
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(display_frame, "Motion Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Calculate simple score based on area
                if is_anomaly:
                    anomaly_score = min(max_area / (self.display_size * self.display_size / 10), 1.0)
                
                # Add overlays
                self._add_overlays(display_frame, is_anomaly, anomaly_score, frame_count)
                
                frame_count += 1
                
                # Encode
                ret, buffer = cv2.imencode('.jpg', display_frame)
                if not ret:
                    print("[ERROR] Failed to encode frame.")
                    continue
                    
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                       
        except Exception as e:
            print(f"[ERROR] Exception in generate_frames: {e}")
        finally:
            print("[DEBUG] Releasing camera.")
            cap.release()

    def _add_overlays(self, frame, is_anomaly, score, frame_count):
        # Status
        color = (0, 0, 255) if is_anomaly else (0, 255, 0)
        status = "ANOMALY (Motion)" if is_anomaly else "Normal"
        
        cv2.rectangle(frame, (0, 0), (self.display_size-1, self.display_size-1), color, 4 if is_anomaly else 2)
        
        cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Score: {score:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Mode: OpenCV MOG2", (10, self.display_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

def run_webcam_detection(autoencoder, encoder, camera_id=0):
    """
    Legacy entry point. Now we recommend using OpenCVAnomalyDetector directly.
    """
    detector = WebcamAnomalyDetector(autoencoder, encoder)
    detector.run_live_detection(camera_id=camera_id, auto_calibrate=True)
