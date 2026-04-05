# 📋 Project Coordination & Development Roadmap
## AI Video Anomaly Detection System (KIET CSM TEAM-11)

This document outlines the architecture, module division, and individual responsibilities for the 5-member project team.

---

## 👥 Team Roles & Responsibilities

The project is divided into **three core disciplines**: Machine Learning (ML), Backend (Flask), and Frontend (UI), plus cross-functional roles for data and testing.

### 1. 🧠 ML Lead (Machine Learning & Core Logic)
**Primary Responsibility:** The "Brain" of the project.
- **Key Modules**: `feature_extractor.py`, `anomaly_detector.py`.
- **Primary Tasks**:
    - Architect and fine-tune the **CNN Autoencoder** (Latent dimension, filter sizes).
    - Implement the **Reconstruction Error** logic (MSE calculation).
    - Develop the **Adaptive Thresholding** algorithm ($\mu + 3\sigma$).
    - Research and integrate different loss functions for better anomaly sensitivity.

### 2. 🌐 Web/UI Lead (Flask Backend & Dashboard Interface)
**Primary Responsibility:** The "Face" and "Vessel" of the project.
- **Key Modules**: `app.py`, `templates/index.html`, `static/css`.
- **Primary Tasks**:
    - Manage the **Flask routing** and session handling.
    - Build a responsive, user-friendly **Surveillance Dashboard**.
    - Develop the **Live Video Feed** streaming using `multipart/x-mixed-replace`.
    - Implement file upload security and storage management.

### 3. 📹 Video Integration & Integration Specialist
**Primary Responsibility:** The "Eyes" and "Nervous System".
- **Key Modules**: `webcam_detector.py`, `video_visualizer.py`.
- **Primary Tasks**:
    - Optimize **OpenCV frame acquisition** for real-time performance (FPS management).
    - Design the **UI Overlays** (Red/Green borders, text annotations on frames).
    - Bridge the data flow between processed ML outputs and the Flask server.
    - Implement the **Video Export** feature (saving processed results with annotations).

### 4. 📊 Data Engineer & Dataset Specialist
**Primary Responsibility:** Quality Control & Preprocessing.
- **Key Focus**: Input Pipelining & Benchmarking.
- **Primary Tasks**:
    - Develop **Preprocessing Scripts** (Resize, RGB normalization, Grayscale options).
    - Curate "Normal" and "Anomalous" test video samples to benchmark accuracy.
    - Conduct **Latency Analysis**: Identifying bottleneck's in the pipeline.
    - Manage the `uploads/` and `static/outputs/` directory structure and cleanup.

### 5. 🛡️ Support & Documentation Specialist
**Primary Responsibility:** Reliability & User Experience.
- **Key Focus**: Testing, Documentation, & Deployment.
- **Primary Tasks**:
    - Lead the **Git Workflow** (Branching, Merge Requests, Commit standards).
    - Technical Writing: Maintain the `README.md` and this `PROJECT_INFO.md`.
    - Conduct **Unit Testing** for individual modules and **Integrative Testing** for the whole app.
    - Manage environment setup (Virtual environments, `requirements.txt`).

---

## 🗓️ Phase-wise Development Plan

### Phase 1: Core ML Development (ML Lead, Data Engineer)
- Finalize Autoencoder architecture.
- Baseline "Normal" data training (or pre-configuration).
- Thresholding logic validation.

### Phase 2: System Architecture (Web Lead, Integration Specialist)
- Setup Flask server structure.
- Integrate OpenCV webcam feed with Flask.
- Build the initial "Anomaly Overlay" visualization.

### Phase 3: Integration & Optimization (Whole Team)
- Connect ML predictions to the Live Dashboard.
- Optimize for high FPS (Performance tuning).
- Implement the search/history sidebar.

### Phase 4: Final Polish & Delivery (Documentation Lead, Web Lead)
- Final UI styling & Responsive design.
- Comprehensive testing.
- Project report & Video demo preparation.

---

## 🛠️ Communication Protocol
- **Git Branching**: Develop on feature branches (`feat/ml-optimization`, `feat/ui-overhaul`).
- **Commits**: Clear, modular commit messages (e.g., `feat: integrate webcam with flask stream`).
- **Documentation**: All new functions must include docstrings for teammate clarity.
