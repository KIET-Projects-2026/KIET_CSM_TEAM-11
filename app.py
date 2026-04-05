from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, Response
import os
import numpy as np
from werkzeug.utils import secure_filename
from main import run_feature_extraction_pipeline, compute_reconstruction_error, detect_anomalies, compute_anomaly_scores
from video_visualizer import save_video_with_anomalies
from feature_extractor import build_autoencoder
from webcam_detector import WebcamAnomalyDetector

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS





from webcam_detector import OpenCVAnomalyDetector

@app.route('/video_feed')
def video_feed():
    # Use the lightweight OpenCV detector instead of the heavy DL model
    detector = OpenCVAnomalyDetector()
    return Response(detector.generate_frames(camera_id=0),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    output_video_url = None
    uploaded_video_url = None
    # Load history from session
    history = session.get('history', [])
    if request.method == 'POST':
        if 'video' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['video']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_video_url = url_for('static', filename=f'../uploads/{filename}')
            try:
                import cv2
                cap = cv2.VideoCapture(filepath)
                input_fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                print(f"[DEBUG] Input video FPS: {input_fps}")
                # Validate FPS: fallback to 25 if invalid
                try:
                    fps_val = float(input_fps)
                    if fps_val < 1 or fps_val > 120 or np.isnan(fps_val):
                        fps_val = 25.0
                except Exception:
                    fps_val = 25.0
                frames, features, encoder, autoencoder = run_feature_extraction_pipeline(filepath, frame_skip=1)
                errors = compute_reconstruction_error(autoencoder, frames)
                anomaly_predictions, threshold = detect_anomalies(errors, percentile=95)
                anomaly_scores = compute_anomaly_scores(errors)
                anomaly_count = int(anomaly_predictions.sum())
                result = f"Anomalies detected: {anomaly_count} / {len(frames)} (Threshold: {threshold:.6f})"
                # Save output video with overlays, matching input FPS (or fallback)
                output_basename = os.path.splitext(filename)[0] + '_output.mp4'
                output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_basename)
                save_video_with_anomalies(frames, anomaly_predictions, output_path, scale_factor=4, fps=int(fps_val))
                output_video_url = url_for('static', filename=f'outputs/{output_basename}')
                # Add to history
                history.insert(0, {
                    'filename': filename,
                    'output_video_url': output_video_url,
                    'anomaly_count': anomaly_count,
                    'total_frames': len(frames),
                    'threshold': f"{threshold:.6f}"
                })
                # Limit history to last 10
                history = history[:10]
                session['history'] = history
            except Exception as e:
                result = f"Error processing video: {str(e)}"
                print(f"Error: {e}")
        else:
            flash('Invalid file type')
            return redirect(request.url)
    return render_template('index.html', result=result, output_video_url=output_video_url, uploaded_video_url=uploaded_video_url, history=history)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
