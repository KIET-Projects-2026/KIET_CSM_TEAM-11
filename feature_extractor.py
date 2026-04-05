"""
Video Feature Extraction Module

This module handles:
- Video loading and frame extraction
- Frame preprocessing (resize, normalize)
- CNN Autoencoder construction
- Deep feature extraction from video frames

NO anomaly detection, NO classification - pure feature extraction.
"""

import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def load_video_frames(video_path, frame_skip=5):
    """
    Load frames from a video file at fixed intervals.
    
    Args:
        video_path (str): Path to the video file
        frame_skip (int): Extract every Nth frame (default: 5)
    
    Returns:
        np.ndarray: Preprocessed frames with shape (num_frames, 128, 128, 3)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Extract every Nth frame
        if frame_count % frame_skip == 0:
            # Resize to (128, 128)
            resized = cv2.resize(frame, (128, 128))
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            normalized = rgb_frame.astype(np.float32) / 255.0
            
            frames.append(normalized)
        
        frame_count += 1
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError("No frames extracted from video")
    
    return np.array(frames)


def build_autoencoder(input_shape=(128, 128, 3), latent_dim=256):
    """
    Build a CNN Autoencoder for feature extraction.
    
    Args:
        input_shape (tuple): Shape of input frames (default: (128, 128, 3))
        latent_dim (int): Dimension of latent feature vector (default: 256)
    
    Returns:
        tuple: (autoencoder_model, encoder_model)
    """
    
    # ========== ENCODER ==========
    encoder_input = keras.Input(shape=input_shape, name='encoder_input')
    
    # Conv Block 1: 128x128x3 -> 64x64x32
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Conv Block 2: 64x64x32 -> 32x32x64
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Conv Block 3: 32x32x64 -> 16x16x128
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Conv Block 4: 16x16x128 -> 8x8x256
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Flatten and create latent representation
    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, activation='relu', name='latent_vector')(x)
    
    # Encoder model
    encoder = keras.Model(encoder_input, latent, name='encoder')
    
    # ========== DECODER ==========
    decoder_input = keras.Input(shape=(latent_dim,), name='decoder_input')
    
    # Reshape for deconvolution
    x = layers.Dense(8 * 8 * 256, activation='relu')(decoder_input)
    x = layers.Reshape((8, 8, 256))(x)
    
    # Deconv Block 1: 8x8x256 -> 16x16x128
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=2, padding='same')(x)
    
    # Deconv Block 2: 16x16x128 -> 32x32x64
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same')(x)
    
    # Deconv Block 3: 32x32x64 -> 64x64x32
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same')(x)
    
    # Deconv Block 4: 64x64x32 -> 128x128x3
    decoder_output = layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', 
                                           strides=2, padding='same', 
                                           name='decoder_output')(x)
    
    # Decoder model
    decoder = keras.Model(decoder_input, decoder_output, name='decoder')
    
    # ========== FULL AUTOENCODER ==========
    autoencoder_output = decoder(encoder(encoder_input))
    autoencoder = keras.Model(encoder_input, autoencoder_output, name='autoencoder')
    
    # Compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return autoencoder, encoder


def extract_video_features(encoder, frames):
    """
    Extract deep features from video frames using the encoder.
    
    Args:
        encoder (keras.Model): Trained or untrained encoder model
        frames (np.ndarray): Preprocessed frames (N, 128, 128, 3)
    
    Returns:
        np.ndarray: Feature vectors (N, latent_dim)
    """
    features = encoder.predict(frames, verbose=0)
    return features