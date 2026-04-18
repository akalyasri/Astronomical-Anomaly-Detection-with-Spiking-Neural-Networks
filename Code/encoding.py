import numpy as np

def rate_encode(signal, threshold=0.05):
    signal = np.array(signal)

    delta = np.diff(signal, prepend=signal[0])

    spikes = np.zeros_like(delta, dtype=float)
    spikes[delta > threshold] = 1.0
    spikes[delta < -threshold] = -1.0

    return spikes