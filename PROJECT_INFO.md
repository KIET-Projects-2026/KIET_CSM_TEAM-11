# 📋 Detailed Project Blueprint & Team Performance Log
## AI Video Anomaly Detection System (KIET CSM TEAM-11)

This document provides a granular view of the project's internal mechanics, specific technical tasks completed by each team member, and the technology-to-member mapping.

---

## 🛠️ Technical Execution & Tasks

### 1. 🧠 ML Lead (Machine Learning Specialist)
**Tech Stack**: `TensorFlow`, `Keras`, `NumPy`
- **Completed Tasks**:
    - **Architecture Design**: Constructed a 4-layer Convolutional Encoder to compress 128x128 RGB images into a 256-dimensional latent space.
    - **Reconstruction Algorithm**: Implemented the `feature_extractor.py` logic where input frames are passed through the autoencoder to generate an output attempt.
    - **Anomaly Calculus**: Coded the `anomaly_detector.py` module to compute Mean Squared Error (MSE) between original pixels and reconstructed pixels.
    - **Statistical Boundary**: Developed the logic for 3-sigma ($\mu \pm 3\sigma$) adaptive thresholding to distinguish anomalies from noise.

### 2. 🌐 Web Lead (Backend & UI Architect)
**Tech Stack**: `Flask`, `HTML5`, `CSS3 (Vanilla)`, `JavaScript`
- **Completed Tasks**:
    - **Server Backbone**: Architected `app.py` to handle asynchronous video requests and webcam streaming.
    - **Surveillance UI**: Designed the main dashboard with a "History" sidebar and real-time status indicators (Normal vs. Anomaly).
    - **Video Streaming**: Implemented the live webcam socketing (MJPEG over HTTP) for lag-free browser visualization.
    - **Session Persistence**: Enabled browser session storage to keep track of previous anomaly detection results.

### 3. 📹 Integration Specialist (Computer Vision Engineer)
**Tech Stack**: `OpenCV (cv2)`, `Python (Thread Management)`
- **Completed Tasks**:
    - **Real-time Pipeline**: Developed `webcam_detector.py` to synchronize camera hardware with ML model inference.
    - **Visual Overlays**: Created the dynamic frame-annotation system (drawing Red/Green borders and error metrics on the screen).
    - **Video Storage**: Built the saving mechanism in `video_visualizer.py` that encodes processed results into MP4 containers.
    - **UI Optimization**: Tuned the frame buffer to maintain stable FPS during simultaneous ML processing and streaming.

### 4. 📊 Data Engineer (Preprocessing & QA)
**Tech Stack**: `NumPy`, `Flask-Uploads`, `FFmpeg`
- **Completed Tasks**:
    - **Input Normalization**: Standardized frame preprocessing (resizing to 128x128 and normalizing pixel values to [0,1]).
    - **Dataset Validation**: Curated local video samples for testing the threshold sensitivity across different lighting environments.
    - **File Management**: Optimized the `uploads/` to `static/outputs/` directory pipeline to handle secure filename saving.
    - **Benchmarking**: Generated performance logs for the ML Lead to verify model accuracy against known anomalous activities.

### 5. 🛡️ Support & DevOps (Documentation & Workflow)
**Tech Stack**: `Git/GitHub`, `Markdown`, `Python venv`
- **Completed Tasks**:
    - **Git Management**: Set up the repository structure, resolved merge conflicts, and managed the `main` branch pushes.
    - **Environment Sync**: Created `requirements.txt` and managed dependencies to ensure the project runs across all team machines.
    - **Technical Documentation**: Authored the premium `README.md` and this `PROJECT_INFO.md` guide.
    - **Bug Tracking**: Performed end-to-end testing of the webcam "Recalibrate" feature and the file-detect-history cycle.

---

## 🔄 System Architecture (How it Works)

The project operates as a **Sequential Pipeline**:

1.  **Ingestion**: `webcam_detector.py` (Integration Specialist) grabs a frame at ~15-30 FPS.
2.  **Preprocessing**: `feature_extractor.py` (Data Engineer) resizes and normalizes the pixel data.
3.  **Inference**: `feature_extractor.py` (ML Lead) passes the frame through the **CNN Autoencoder**.
4.  **Comparison**: `anomaly_detector.py` (ML Lead) compares the Original Frame vs. Reconstructed Frame.
5.  **Classification**: If $MSE > Threshold$, the frame is flagged as an **ANOMALY**.
6.  **Visualization**: `video_visualizer.py` (Integration Specialist) draws the Red border and metrics.
7.  **Delivery**: `app.py` (Web Lead) streams the final result to the browser dashboard.

---

## 🗺️ Tech Usage Matrix

| Member | Primary Tool | Secondary Tool | Domain |
| :--- | :--- | :--- | :--- |
| **ML Lead** | TensorFlow | NumPy | Neural Networks |
| **Web Lead** | Flask | CSS/JS | System Interface |
| **Integration** | OpenCV (cv2) | Multiprocessing | Real-time Vision |
| **Data Eng.** | NumPy | OS/Filesystem | Preprocessing |
| **Support** | Git | Markdown | CI/CD & Docs |
