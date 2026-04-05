# 📖 Line-by-Line Code Explanation
## AI Video Anomaly Detection System (KIET CSM TEAM-11)

This document provides a comprehensive walkthrough of the project's source code, explaining the logic and purpose of every major component.

---

## 1. `feature_extractor.py` (The AI Brain)
This file defines the **CNN Autoencoder** architecture.

| Lines | Purpose | Explanation |
| :--- | :--- | :--- |
| **19-29** | `load_video_frames` | Function to load and preprocess video frames. |
| **30** | `cv2.VideoCapture` | Opens the video file using the OpenCV library. |
| **38-42** | `cap.read()` | A loop that reads the video frame-by-frame until the end. |
| **45** | `frame_count % frame_skip` | Optimization: Only extract every Nth frame to save memory/time. |
| **47** | `cv2.resize(frame, (128, 128))` | Resizes the frame to a standard 128x128 pixel square for the AI. |
| **53** | `/ 255.0` | **Normalization**: Converts pixel values (0-255) to 0.0-1.0 float for faster ML training. |
| **67-77** | `build_autoencoder` | Constructor for the Neural Network. |
| **83-96** | **Encoder Layers** | 4 Convolutional layers + MaxPooling. Each layer "shrinks" the image but "increases" the feature depth (filters go from 32 to 256). |
| **106-127** | **Decoder Layers** | 4 Transposed Convolutional layers. This "reconstructs" the 128x128 image back from the 256-dimensional latent code. |
| **130** | `decoder(encoder(input))` | Connects the Encoder and Decoder to form the full "Autoencoder" pipeline. |
| **134** | `compile(optimizer='adam', loss='mse')` | Sets up the learning rule (Adam) and the goal (minimize Mean Squared Error). |

---

## 2. `anomaly_detector.py` (The Mathematical Evaluator)
This file handles the comparison and scoring.

| Lines | Purpose | Explanation |
| :--- | :--- | :--- |
| **14-25** | `compute_reconstruction_error` | Predicts what the AI "thinks" the frame should look like. |
| **28** | `abs(original - predicted)` | Calculates the difference between the actual video and the AI's reconstruction. |
| **29** | `np.mean(..., axis=(1,2,3))` | Computes the **Mean Squared Error (MSE)**. This single number represents how "weird" the frame is to the AI. |
| **45-55** | `detect_anomalies` | Decides if a frame is an anomaly. |
| **61** | `percentile(errors, percentile)` | In Video Mode: Flags the top X% highest-error frames as anomalies. |
| **78-85** | `compute_anomaly_scores` | **Normalization**: Scales the error values to a 0-100 "Anomaly Score" for the user interface. |

---

## 3. `app.py` (The Web Server)
This file integrates everything into a browser-based dashboard.

| Lines | Purpose | Explanation |
| :--- | :--- | :--- |
| **14** | `app = Flask(__name__)` | Initializes the web server framework. |
| **33-38** | `@app.route('/video_feed')` | The endpoint that streams live webcam frames to the browser. |
| **41-100** | `def index()` | The main dashboard logic. It handles both the GET (viewing) and POST (uploading videos) requests. |
| **58** | `file.save(filepath)` | Saves the user's uploaded security footage to the `uploads/` folder. |
| **73** | `run_feature_extraction_pipeline` | Triggers the ML logic on the uploaded video. |
| **82** | `save_video_with_anomalies` | Generates a new version of the video with **Red Borders** around detected anomalies. |
| **85-94** | `history.insert(...)` | Saves the results to the `session` object so the user can see past detections in the sidebar. |

---

## 4. `webcam_detector.py` (Live Surveillance Optimizor)
This file handles the complexity of "moving" live data through the AI.

| Lines | Purpose | Explanation |
| :--- | :--- | :--- |
| **15-28** | `OpenCVAnomalyDetector` | A lightweight class for real-time monitoring without heavy DL overhead if needed. |
| **34-58** | `generate_frames` | A generator function (uses `yield`) that continuously grabs frames from the camera. |
| **45** | `baseline_subtraction` | Detects motion by comparing the current frame to the previous one. |
| **47** | `cv2.threshold` | Converts gray motion into a black-and-white mask. |
| **55-65** | `cv2.imencode('.jpg', frame)` | Converts the raw frame into a JPG format so the browser can display it as a video stream. |

---

## 5. `video_visualizer.py` (The Output Generator)
This file creates the user-facing "evidence" videos.

| Lines | Purpose | Explanation |
| :--- | :--- | :--- |
| **20-45** | `draw_anomaly_overlay` | Drawing function. |
| **22** | `cv2.rectangle(...)` | Draws a **Green border** for normal frames and a **Red border** for anomalies. |
| **33** | `cv2.putText(...)` | Stamps the "Anomaly Score" and "Status" text onto the video frame. |
| **50-80** | `save_video_with_anomalies` | Uses `cv2.VideoWriter` to create a finished MP4 file containing the AI's markings. |

---

## 6. `main.py` (The Control Center)
The entry point for using the project via command-line.

| Lines | Purpose | Explanation |
| :--- | :--- | :--- |
| **120-150** | `argparse` | Allows users to run the project with options like `--mode webcam` or `--video sample.mp4`. |
| **160-200** | `main()` | The master controller that decides whether to open the camera or process a file. |
| **210** | `detect_live()` | High-accuracy mode: Runs the full CNN Autoencoder on every single webcam frame. |
