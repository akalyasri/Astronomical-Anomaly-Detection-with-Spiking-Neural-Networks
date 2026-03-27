import numpy as np

# convert continuous signal to spike train using deterministic thresholding
# high spike: 1, low spike: 0
def rate_encode(signal, threshold=0.5):
    signal = np.array(signal)
    
    # scale to [0,1] first
    signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
    
    spikes = (signal > threshold).astype(float)
    return spikes