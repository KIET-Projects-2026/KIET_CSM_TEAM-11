# Video Anomaly Detection System

Real-time anomaly detection for video files and live webcam feeds using deep learning autoencoders.

## Features

🎥 **Dual Mode Operation**
- **Video File Mode**: Analyze pre-recorded videos for anomalies
- **Webcam Mode**: Real-time anomaly detection from live camera feed

🧠 **Deep Learning Pipeline**
- CNN Autoencoder for feature extraction and reconstruction
- Adaptive thresholding with rolling statistics
- Automatic calibration for baseline behavior

📊 **Visualization**
- Live video playback with anomaly overlays
- Color-coded borders (green = normal, red = anomaly)
- Real-time metrics (error, threshold, anomaly score)
- Interactive controls (pause, recalibrate, navigate)

## Requirements

- Python 3.8+
- Webcam (for live detection mode)
- GPU recommended but not required

## Installation

### 1. Clone or Download the Project

```bash
cd /path/to/project
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install opencv-python numpy tensorflow
```

**Dependencies:**
- `opencv-python` - Video and webcam capture, visualization
- `numpy` - Numerical operations
- `tensorflow` - Deep learning autoencoder model

## Project Structure

```
project/
├── main.py                 # Main entry point with CLI
├── feature_extractor.py    # Video loading & autoencoder
├── anomaly_detector.py     # Anomaly detection logic
├── video_visualizer.py     # Video playback visualization
├── webcam_detector.py      # Live webcam detection
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Usage

### Quick Start - Webcam Mode

```bash
python main.py --mode webcam
```

This will:
1. Build the autoencoder model (~2-3 seconds)
2. Open your default webcam
3. Auto-calibrate by observing 50 normal frames
4. Start real-time anomaly detection

### Video File Mode

```bash
# Analyze a video file
python main.py --mode video --video path/to/your/video.mp4
```

### Advanced Options

```bash
# Use specific camera
python main.py --mode webcam --camera 1

# Custom latent dimension
python main.py --mode webcam --latent-dim 512

# Video file with custom settings
python main.py --mode video --video sample.mp4 --latent-dim 128
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | `video` | Detection mode: `video` or `webcam` |
| `--video` | str | `sample_video.mp4` | Path to video file (video mode) |
| `--camera` | int | `0` | Camera device ID (webcam mode) |
| `--latent-dim` | int | `256` | Autoencoder latent dimension |

### Interactive Controls

**Webcam Mode:**
- `ESC` or `q` - Quit detection
- `SPACE` or `p` - Pause/Resume
- `c` - Recalibrate with new baseline

**Video Mode:**
- `ESC` or `q` - Quit playback
- `SPACE` or `p` - Pause/Resume
- Arrow keys - Navigate frames

## How It Works

### Architecture

1. **CNN Autoencoder**
   - Encoder: Compresses 128×128 RGB frames to latent vectors
   - Decoder: Reconstructs frames from latent representation
   - Architecture: 4 conv layers → latent (256D) → 4 deconv layers

2. **Anomaly Detection**
   - Reconstruction error = MSE(original, reconstructed)
   - Threshold = μ + 3σ (adaptive in webcam mode)
   - Anomaly detected when error > threshold

3. **Webcam Calibration**
   - Collects 50 baseline frames of normal activity
   - Computes mean (μ) and standard deviation (σ)
   - Sets initial threshold for anomaly detection
   - Continuously updates statistics using rolling window

### Pipeline Flow

```
Input Frame → Preprocess (resize, normalize) 
           → Autoencoder (encode + decode)
           → Compute Reconstruction Error
           → Compare with Threshold
           → Classify (Normal / Anomaly)
           → Visualize with Overlays
```

## Example Workflows

### 1. Analyze a Video File

```bash
# Process your video
python main.py --mode video --video my_video.mp4

# Output: Interactive visualization window
# - Green border = normal frames
# - Red border = anomaly frames
# - Press SPACE to pause/play
# - Press ESC to exit
```

### 2. Live Webcam Monitoring

```bash
# Start webcam detection
python main.py --mode webcam

# Calibration phase (2 seconds + 50 frames)
# → Shows "NOT CALIBRATED" in yellow

# Detection phase
# → Normal activity: Green border
# → Anomaly detected: Red flashing border

# Press 'c' anytime to recalibrate
```

### 3. Creating Anomalies for Testing

Try these in webcam mode:
- Move unusual objects into view
- Make sudden lighting changes
- Perform unexpected gestures
- Block/unblock camera rapidly
- Introduce new people or objects

## Troubleshooting

### Webcam Not Opening

```bash
# Test different camera IDs
python main.py --mode webcam --camera 0
python main.py --mode webcam --camera 1
```

### Video File Not Found

```bash
# Use absolute path
python main.py --mode video --video /full/path/to/video.mp4
```

### Low FPS / Slow Performance

- Reduce latent dimension: `--latent-dim 128`
- Close other applications
- Use GPU if available (CUDA-enabled TensorFlow)

### Too Many/Few Anomalies Detected

**Webcam Mode:**
- Press `c` to recalibrate if environment changed
- Ensure consistent lighting during calibration
- Keep normal activity during calibration phase

**Video Mode:**
- Anomaly threshold is set at 95th percentile
- Modify in `main.py` line 262: `percentile=95` (increase = fewer anomalies)

## Performance Notes

- **Autoencoder Build Time**: 2-3 seconds (one-time at startup)
- **Calibration Time**: ~5 seconds (50 frames at 10 FPS)
- **Real-time FPS**: 10-15 FPS (CPU), 30+ FPS (GPU)
- **Memory Usage**: ~500 MB (model + buffers)

## Model Architecture Details

```
Encoder:
  Conv2D(32) → MaxPool → Conv2D(64) → MaxPool 
  → Conv2D(128) → MaxPool → Conv2D(256) → MaxPool
  → Flatten → Dense(latent_dim)

Decoder:
  Dense(8×8×256) → Reshape
  → ConvTranspose(128) → ConvTranspose(64) 
  → ConvTranspose(32) → ConvTranspose(3)

Total Parameters: ~9.2M (latent_dim=256)
```

## Contributing

Feel free to extend this project:
- Add different anomaly detection algorithms
- Implement training mode for custom datasets
- Add audio anomaly detection
- Export detection results to file
- Create web interface

## License

Free to use for educational and research purposes.

## Acknowledgments

Built with:
- TensorFlow/Keras for deep learning
- OpenCV for video processing
- NumPy for numerical operations

---

**Need Help?** Check the command-line help:
```bash
python main.py --help
```

