import argparse
import librosa
import numpy as np
import cv2
import soundfile as sf
from scipy.signal import butter, filtfilt
from pathlib import Path

def butter_lowpass(cutoff, fs, order=5):
    """
    Design a Butterworth low-pass filter.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff, fs, order=1):
    """
    Apply a Butterworth low-pass filter to a signal.
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def spectral_subtraction(S, noise_estimate):
    """
    Apply spectral subtraction to reduce noise.
    """
    S_cleaned = np.maximum(S - noise_estimate[:, np.newaxis], 0)
    return S_cleaned

def estimate_noise(S, noise_floor_percent=0.1):
    """
    Estimate the noise from the spectrogram by averaging the lowest percent of frames.
    """
    num_frames = S.shape[1]
    num_noise_frames = max(1, int(noise_floor_percent * num_frames))
    noise_estimate = np.mean(np.sort(S, axis=1)[:, :num_noise_frames], axis=1)
    return noise_estimate

def spectrogram_to_audio(spectrogram_path, save_path, sr=22050, n_fft=2048, hop_length=512, num_iters=1000):
    """
    Convert a spectrogram image back to an audio file with enhanced quality.
    """
    # Load the spectrogram image
    S_img = cv2.imread(spectrogram_path, cv2.IMREAD_GRAYSCALE)

    # Convert image to float32 and scale back to original amplitude range
    S = cv2.normalize(S_img, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
    S = librosa.db_to_amplitude(S * 80.0 - 80.0)  # Adjusting the range to match typical dB scale

    # Estimate noise and perform spectral subtraction
    noise_estimate = estimate_noise(S)
    S_cleaned = spectral_subtraction(S, noise_estimate)

    # Use Griffin-Lim algorithm for better phase estimation
    y = librosa.griffinlim(S_cleaned, n_iter=num_iters, hop_length=hop_length, win_length=n_fft)

    # Apply a low-pass filter to reduce high-frequency noise
    y = apply_lowpass_filter(y, cutoff=sr // 4, fs=sr)  # Adjusted cutoff frequency to avoid errors

    # Save the recovered audio to a file using soundfile
    sf.write(save_path, y, sr)
    print(f"Audio saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert spectrogram to audio")
    parser.add_argument('--spectrogram', type=str, required=True, help="Path to the spectrogram image")
    parser.add_argument('--output', type=str, required=True, help="Path to save the output audio file")
    parser.add_argument('--sample_rate', type=int, default=22050, help="Sample rate for the audio file")
    parser.add_argument('--n_fft', type=int, default=2048, help="FFT window size")
    parser.add_argument('--hop_length', type=int, default=512, help="Number of samples between successive frames")
    parser.add_argument('--num_iters', type=int, default=1000, help="Number of iterations for the Griffin-Lim algorithm")
    args = parser.parse_args()

    spectrogram_to_audio(args.spectrogram, args.output, args.sample_rate, args.n_fft, args.hop_length, args.num_iters)
