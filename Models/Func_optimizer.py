import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torchmetrics.audio import ComplexScaleInvariantSignalNoiseRatio
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import os
import re
import joblib

device = "cuda" if torch.cuda.is_available() else "cpu"

sisnr = ComplexScaleInvariantSignalNoiseRatio().to(device)

def get_files(directory):
    files = []
    all_files = sorted(os.listdir(directory))
    for file in all_files:
        if os.path.isfile(os.path.join(directory, file)):
            if '.jams' not in file and '.txt' not in file:
                full_path = os.path.join(directory, file)
                full_path = full_path.replace("\\", "/")
                files.append(full_path)
    
    print(len(files))
        
    files = sorted(files, key = lambda x : int(os.path.basename(x).split('_')[1].split('.')[0]))

    return files
  
files1 = get_files('/kaggle/input/people-fsd/Mixed2/Mixed2')  # Add your dataset path here
files2 = get_files('/kaggle/input/people-fsd/Mixed3/Mixed3')  # Add your dataset path here
files3 = get_files('/kaggle/input/people-fsd/ground_truth/ground_truth') # Add your dataset path here

def gradient_descent(mixed_stft, bg_stft, clean_stft, alpha_init=1.0, lr=0.01, epochs=100):
    alpha = torch.tensor(alpha_init, dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.SGD([alpha], lr=lr)
    
    for epoch in range(epochs):
        # Perform spectral subtraction: subtracted_stft = mixed_stft - alpha * bg_stft
        subtracted_stft = mixed_stft - alpha * bg_stft
        
        # Compute the SI-SNR between the cleaned signal and the target clean signal
        loss = -sisnr(clean_stft, subtracted_stft)
        
        # Negative SI-SNR (since we want to maximize SI-SNR)
        
        # Zero gradients from previous step
        optimizer.zero_grad()
        
        # Compute gradients
        loss.backward()
        
        # Update alpha using gradient descent
        optimizer.step()
        
        # Print loss and alpha value every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Alpha: {alpha.item():.4f}')
    
    return alpha

# Example function to generate training data
def generate_training_data(num_samples):
    alphas = []
    features = []
    
    for i in range(num_samples):
        # Generate random mixed, background, and clean signals
        mixed_signal, _ = torchaudio.load(files1[i])
        bg_signal, _  = torchaudio.load(files3[i])
        clean_signal, _ = torchaudio.load(files2[i])

        mixed_signal = mixed_signal.to(device)
        bg_signal = bg_signal.to(device)
        clean_signal = clean_signal.to(device)
        
        # Compute STFTs
        n_fft = 1024
        hop_length = 512
        mixed_stft = torch.stft(mixed_signal, n_fft=n_fft, hop_length=hop_length, return_complex=True)
        bg_stft = torch.stft(bg_signal, n_fft=n_fft, hop_length=hop_length, return_complex=True)
        clean_stft = torch.stft(clean_signal, n_fft=n_fft, hop_length=hop_length, return_complex=True)
        
        # Compute optimal alpha
        alpha = gradient_descent(mixed_stft, bg_stft, clean_stft).to(device)  # Function as described earlier
        
        # Extract features (e.g., magnitudes)
        mixed_mag = torch.abs(mixed_stft).mean().item()
        bg_mag = torch.abs(bg_stft).mean().item()
        
        # Store features and target alpha
        features.append([mixed_mag, bg_mag])
        alphas.append(alpha.item())

        if i % 1000 == 0 and i != 0: 
            print("Done")
    
    return np.array(features), np.array(alphas)

# Generate dataset
features, alphas = generate_training_data()

X_train, X_test, y_train, y_test = train_test_split(features, alphas, test_size=0.2, random_state=42)

# Train a regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Save the model
joblib.dump(regressor, "alpha_estimator.pkl")

