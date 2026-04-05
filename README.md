# 🛡️ AI Video Anomaly Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-Web-lightgrey.svg)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer--Vision-green.svg)](https://opencv.org/)

An advanced, real-time security and surveillance intelligence system that detects unusual activities in video streams using **Deep Learning Autoencoders**. 

This system automatically learns "normal" patterns and flags anything that deviates significantly, providing a proactive approach to monitoring without needing pre-labeled "bad" data.

---

## 🚀 Quick Start (Run the Web App)

The project includes a modern **Flask-based Web Interface** for easy interaction.

### 1. Prerequisites
- Python 3.12.6 (Recommended)
- Webcam (for Live Detection)

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/KIET-Projects-2026/KIET_CSM_TEAM-11.git
cd KIET_CSM_TEAM-11

# Create and Activate Virtual Environment
python -m venv .venv
.\.venv\Scripts\activate

# Install Core Dependencies
pip install flask opencv-python numpy tensorflow werkzeug
```

### 3. Launching
```bash
python app.py
```
Visit **`http://127.0.0.1:5000`** in your browser to start.

---

## 🛠️ Technology Stack & Core Tools

| Technology | Purpose | Implementation Detail |
|:--- |:--- |:--- |
| **TensorFlow / Keras** | **Brain** | CNN Autoencoder architecture for spatial feature learning. |
| **OpenCV** | **Eyes** | Efficient video frame acquisition, resizing, and live visualization overlays. |
| **Flask** | **Vessel** | Lightweight web server to host the dashboard and live streaming endpoints. |
| **NumPy** | **Logic** | Matrix operations for MSE (Mean Squared Error) and statistical thresholding. |
| **Werkzeug** | **Security** | Safe file handling for video uploads. |

---

## 🧠 How It Works: The Science Behind

### 1. The Autoencoder Architecture
Instead of training a model to recognize specific crimes (like "theft" or "fire"), we train it to **reconstruct normal environment frames**. 
- **Encoder**: Compresses a 128x128 frame into a small "latent vector" (bottleneck).
- **Decoder**: Attempts to rebuild the original frame from that small vector.

### 2. High Reconstruction Error = Anomaly
When the model sees something it hasn't seen before (e.g., a person running in a restricted area, a sudden object), it **fails to reconstruct it accurately**. 
- **Normal Activity**: Error is low (The model "knows" this scene).
- **Anomaly**: Error is high (The model is "confused" by the new pattern).

### 3. Adaptive Thresholding ($\mu + k\sigma$)
The system uses a mathematical boundary to decide what is "too high":
$$Threshold = \mu + (3 \times \sigma)$$
*Where $\mu$ is the average error and $\sigma$ is the standard deviation.*
- **In Webcam Mode**: It calculates this threshold during a brief 5-second "Calibration Phase".
- **In Video Mode**: It calculates the 95th percentile of errors across the whole file.

---

## 📁 Project Structure

```text
├── app.py                  # Flask Web Controller
├── main.py                 # Core CLI entry point
├── feature_extractor.py    # CNN Autoencoder Definition
├── anomaly_detector.py     # Mathematical core (MSE & Predictions)
├── webcam_detector.py      # Live streaming logic (Optimized)
├── video_visualizer.py     # OpenCV UI components & Video Saving
├── requirements.txt        # Automated setup list
├── static/                 # CSS, JS, and Processed Outputs
└── templates/              # HTML Dashboards
```

---

## 🖱️ User Controls

### Web Interface
- **Upload Video**: Submit MP4/AVI files for batch processing.
- **Start Webcam**: Toggle real-time surveillance.
- **History Sidebar**: Review past detection results and anomaly counts.

### Desktop/CLI Mode (`main.py`)
- `SPACE`: Pause/Resume
- `C`: Recalibrate (updates the "normal" baseline)
- `Q` / `ESC`: Terminate Session

---

## 🛡️ Security & Privacy
- **Local Processing**: All frames are processed locally; no video data is sent to external clouds.
- **Privacy Masking**: The model learns spatial patterns, not identity details.

## 📈 Future Roadmap
- [ ] **Temporal Analysis**: Adding LSTM layers to detect anomalies over time (speed, movement direction).
- [ ] **Multi-Camera Support**: Manage multiple feeds from a single dashboard.
- [ ] **E-mail Alerts**: Automated notifications when a high-score anomaly is detected.

---

### Developed by
**Team KIET_CSM_TEAM-11**  
*Empowering Security through Artificial Intelligence.*
