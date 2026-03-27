import numpy as np
import torch


def normalize(signal):
    mean = np.mean(signal)
    std = np.std(signal)

    if std == 0:
        return signal

    return (signal - mean) / std


# scaling to [0, 1]
def min_max_scale(signal):
    min_val = np.min(signal)
    max_val = np.max(signal)

    if max_val - min_val == 0:
        return signal

    return (signal - min_val) / (max_val - min_val)

# smoothing
def smooth_signal(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')


# preprocessing pipeline
def preprocess(signal, use_smoothing=False):
    if use_smoothing:
        signal = smooth_signal(signal)

    signal = normalize(signal)
    signal = min_max_scale(signal)

    return signal

# converting numpy array to PyTorch tensor
def to_tensor(signal):
    return torch.tensor(signal, dtype=torch.float32).unsqueeze(1)