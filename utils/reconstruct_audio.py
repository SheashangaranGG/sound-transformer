import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

def load_spectrogram(image_file):
    img = plt.imread(image_file)
    return img

def spectrogram_to_audio(spectrogram):
    # Reconstruct the waveform from the spectrogram using Griffin-Lim
    S = librosa.db_to_power(spectrogram)
    y = librosa.feature.inverse.mel_to_audio(S)
    return y

def reconstruct_audio(spectrogram_file, output_filename):
    spectrogram_img = load_spectrogram(spectrogram_file)
    audio = spectrogram_to_audio(spectrogram_img)
    sf.write(output_filename, audio, 22050)
