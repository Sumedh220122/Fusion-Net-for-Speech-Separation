import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import torchaudio
import torch
import os
from scipy.signal import medfilt
import joblib

class Background_Reduction:
    def __init__(self):
        self.mask_floor = 0.1
        self.regressor = joblib.load("Checkpoints/alpha_estimator.pkl")

    def reduction(self, input_audio, background_audio):
        mixture, sr = librosa.load(input_audio , sr=None)
        background, _ = librosa.load(background_audio)

        max_len = max(len(mixture), len(background))
        mixture = np.pad(mixture, (0, max_len - len(mixture)), mode='constant')
        background = np.pad(background, (0, max_len - len(background)), mode='constant')

        for _ in range(4):
            noisy_stft = librosa.stft(mixture, n_fft=1024, hop_length=256, win_length=1024)
            noise_stft = librosa.stft(background, n_fft=1024, hop_length=256, win_length=1024)

            noisy_power = np.abs(noisy_stft) ** 2
            noise_power = np.abs(noise_stft) ** 2

            mask = np.maximum((noisy_power - noise_power) / (noisy_power + noise_power + 1e-10), self.mask_floor)

            enhanced_magnitude = np.abs(noisy_stft) * mask

            enhanced_stft = enhanced_magnitude * np.exp(1j * np.angle(noisy_stft))

            enhanced_signal = librosa.istft(enhanced_stft, hop_length=256, win_length=1024)

            mixture = enhanced_signal

        return enhanced_signal
    
    def compute_snr(self, signal, noise):
        signal_power = torch.mean(signal**2)
        noise_power = torch.mean(noise**2)
        return 10 * torch.log10(signal_power / (noise_power + 1e-6))  # SNR in dB

    
    def spectral_subtraction(self, mixture, background, target_folder, i):
        mixture_path = mixture
        background_path = background

        for j in range(1):
            mixed_waveform, sample_rate = torchaudio.load(mixture_path)
            noise_waveform, _ = torchaudio.load(background_path)

            if noise_waveform.shape[1] > mixed_waveform.shape[1]:
                noise_waveform = noise_waveform[:, :mixed_waveform.shape[1]]
            else:
                noise_waveform = torch.nn.functional.pad(noise_waveform, (0, mixed_waveform.shape[1] - noise_waveform.shape[1]))
                    
            n_fft = 1024 
            hop_length = 512
            stft_transform = torch.stft
            mixed_stft = torch.stft(mixed_waveform, n_fft=n_fft, hop_length=hop_length, return_complex=True)
            noise_stft = torch.stft(noise_waveform, n_fft=n_fft, hop_length=hop_length, return_complex=True)

            # snr = self.compute_snr(mixed_waveform, noise_waveform)
            # if snr > 10:
            #     alpha = 0.5  # Higher SNR, less noise subtraction
            # elif snr > 0:
            #     alpha = 0.8  # Moderate noise
            # else:
            #     alpha = 1.0  # Low SNR, more aggressive subtraction

            mixed_mag = torch.abs(mixed_stft).mean().item()
            bg_mag = torch.abs(noise_stft).mean().item()
            features = np.array([[mixed_mag, bg_mag]])

            alpha = self.regressor.predict(features)[0]
            
            mixed_magnitude, mixed_phase = torch.abs(mixed_stft), torch.angle(mixed_stft)
            noise_magnitude, noise_phase = torch.abs(noise_stft), torch.angle(noise_stft)

            epsilon = 1e-6 
            subtracted_magnitude = torch.relu(mixed_magnitude - alpha * noise_magnitude)  # Using ReLU to avoid negative values
            reconstructed_stft = subtracted_magnitude * torch.exp(1j * mixed_phase)
            
            inverse_transform = torch.istft
            cleaned_waveform = torch.istft(reconstructed_stft, n_fft=n_fft, hop_length=hop_length)

            cleaned_waveform = torch.nn.functional.pad(cleaned_waveform, (0, noise_waveform.shape[1] - cleaned_waveform.shape[1]))

            output_path = os.path.join(target_folder, f'enhanced.wav')
            torchaudio.save(output_path, cleaned_waveform, 44100)

            mixture_path = output_path
