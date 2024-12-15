import argparse
import librosa
import numpy as np
import cv2
from pathlib import Path

def audio_to_spectrogram(audio_path, save_path, duration=None, n_fft=2048, hop_length=512, win_length=None):
    """
    Convert an audio file to a spectrogram and save it as an image.

    Args:
        audio_path (str): The path to the audio file.
        save_path (str): The path to save the spectrogram image.
        duration (int or None): Duration of the audio file to process in seconds (None for full length).
        n_fft (int): FFT window size.
        hop_length (int): Number of samples between successive frames.
        win_length (int): Each frame of audio is windowed by window(). The default value is n_fft.

    Returns:
        None
    """
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, duration=duration)

        # Compute the Short-Time Fourier Transform (STFT)
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        S = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # Normalize the spectrogram to 0-255 for visualization
        S_normalized = cv2.normalize(S, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Convert grayscale spectrogram to RGB
        S_rgb = cv2.cvtColor(S_normalized, cv2.COLOR_GRAY2RGB)

        # Save the spectrogram as a PNG image
        cv2.imwrite(save_path, S_rgb)

        print(f"Spectrogram saved to {save_path}")

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert audio to spectrogram")
    parser.add_argument('--source', type=str, required=True, help="Path to the input audio file")
    parser.add_argument('--output', type=str, required=True, help="Path to save the output spectrogram image")
    parser.add_argument('--duration', type=int, default=None, help="Duration of the audio to process in seconds")
    parser.add_argument('--n_fft', type=int, default=2048, help="FFT window size")
    parser.add_argument('--hop_length', type=int, default=512, help="Number of samples between successive frames")
    parser.add_argument('--win_length', type=int, default=None, help="Window length for each frame")
    args = parser.parse_args()

    audio_to_spectrogram(args.source, args.output, args.duration, args.n_fft, args.hop_length, args.win_length)
