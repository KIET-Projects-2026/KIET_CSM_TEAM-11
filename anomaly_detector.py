"""
Anomaly Detection Module

This module handles:
- Computing reconstruction errors
- Detecting anomalies based on thresholds
- Computing anomaly scores

This is a placeholder module for future anomaly detection implementation.
"""

import numpy as np


def compute_reconstruction_error(autoencoder, frames):
    """
    Compute reconstruction error for each frame.
    
    Args:
        autoencoder: The autoencoder model
        frames (np.ndarray): Original frames (N, 128, 128, 3)
    
    Returns:
        np.ndarray: Reconstruction errors for each frame
    """
    # Reconstruct frames
    reconstructed = autoencoder.predict(frames, verbose=0)
    
    # Compute mean squared error per frame
    errors = np.mean((frames - reconstructed) ** 2, axis=(1, 2, 3))
    
    return errors


def detect_anomalies(errors, threshold=None, percentile=95):
    """
    Detect anomalies based on reconstruction errors.
    
    Args:
        errors (np.ndarray): Reconstruction errors
        threshold (float, optional): Manual threshold. If None, uses percentile
        percentile (int): Percentile for automatic threshold (default: 95)
    
    Returns:
        tuple: (anomaly_predictions, threshold_used)
    """
    if threshold is None:
        threshold = np.percentile(errors, percentile)
    
    anomaly_predictions = (errors > threshold).astype(int)
    
    return anomaly_predictions, threshold


def compute_anomaly_scores(errors):
    """
    Normalize reconstruction errors to anomaly scores in [0, 1].
    
    Args:
        errors (np.ndarray): Reconstruction errors
    
    Returns:
        np.ndarray: Normalized anomaly scores
    """
    # Normalize to [0, 1] range
    min_error = np.min(errors)
    max_error = np.max(errors)
    
    if max_error - min_error == 0:
        return np.zeros_like(errors)
    
    scores = (errors - min_error) / (max_error - min_error)
    
    return scores