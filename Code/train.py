import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from preprocessing import preprocess, to_tensor
from encoding import rate_encode
from model import SimpleSNN

# auto create a results folder
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# synthetic data generation (1 vs. multiple)
def generate_signal(length=500, amplitude=1.0, frequency=1.0, phase=0.0, anomaly=False):
    t = np.linspace(0, 10, length)
    signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    if anomaly:
        start, end = length // 3, length // 2
        signal[start:end] += np.random.uniform(1.5, 2.5)
    return signal

def generate_signals(num_signals=50, anomaly=False):
    signals = []
    for _ in range(num_signals):
        amp = np.random.uniform(0.8, 1.2)
        freq = np.random.uniform(0.8, 1.2)
        phase = np.random.uniform(0, np.pi/4)
        signal = generate_signal(amplitude=amp, frequency=freq, phase=phase, anomaly=anomaly)
        signals.append(signal)
    return signals

# training loop
def train_model(model, optimizer, normal_signals, epochs=50, print_every=5):
    model.train()
    loss_fn = nn.MSELoss()
    all_losses = []

    # preprocess and encode all signals first
    processed_signals = []
    for sig in normal_signals:
        sig_proc = preprocess(sig)
        spikes = rate_encode(sig_proc, threshold=0.05)
        tensor_sig = to_tensor(spikes)
        processed_signals.append(tensor_sig)

    # stacking signals into a batch
    x_batch = torch.stack(processed_signals, dim=1)

    for epoch in range(epochs):
        output = model(x_batch)
        loss = loss_fn(output, x_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_losses.append(loss.item())
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return all_losses, x_batch, output

# anomaly detection
def compute_anomaly_scores(model, signals, labels=None):
    model.eval()
    processed_signals = []
    for sig in signals:
        sig_proc = preprocess(sig)
        spikes = rate_encode(sig_proc, threshold=0.05)
        tensor_sig = to_tensor(spikes)
        processed_signals.append(tensor_sig)

    x_batch = torch.stack(processed_signals, dim=1)
    with torch.no_grad():
        output = model(x_batch)
        mse_per_signal = ((output - x_batch)**2).mean(dim=0)
        scores = mse_per_signal.squeeze().numpy()

    return scores, x_batch, output

# main
if __name__ == "__main__":
    # initializing model and optimizer
    model = SimpleSNN()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # generate training signals (only normal)
    num_train = 50
    normal_signals = generate_signals(num_signals=num_train, anomaly=False)

    # training the model
    epochs = 50
    print_every = 5
    all_losses, x_train, output_train = train_model(model, optimizer, normal_signals, epochs, print_every)

    # plotting training loss
    plt.figure(figsize=(8, 4))
    plt.plot(all_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss on Normal Signals')
    plt.legend()
    # plt.show()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_loss.png"))
    plt.close()

    # generate test signals (mix of normal and anomalous)
    num_test = 20
    normal_test = generate_signals(num_test, anomaly=False)
    anomalous_test = generate_signals(num_test, anomaly=True)
    test_signals = normal_test + anomalous_test
    labels = [0]*num_test + [1]*num_test  # 0=normal, 1=anomaly

    # compute anomaly scores
    scores, x_test, output_test = compute_anomaly_scores(model, test_signals, labels)

    # simple thresholding for anomaly detection
    threshold = np.mean(scores[:num_test]) - 2*np.std(scores[:num_test])
    print(f"Anomaly detection threshold (lower than normal) = {threshold:.4f}\n")

    for i, score in enumerate(scores):
        if score < threshold:
            print(f"Signal {i:02d} flagged as ANOMALY (score = {score:.4f})")
        else:
            print(f"Signal {i:02d} considered NORMAL (score = {score:.4f})")

    # plot example signal vs reconstruction (first anomalous signal)
    idx_example = num_test  # first anomalous
    plt.figure(figsize=(12, 4))
    plt.plot(x_test[:, idx_example, :].numpy(), label='Encoded Singal (Anomalous)')
    plt.plot(output_test[:, idx_example, :].numpy(), label='Reconstructed Signal')
    plt.title('SNN Reconstruction of Anomalous Signal')
    plt.legend()
    #plt.show()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "reconstruction_example.png"))
    plt.close()

    # plot anomaly scores
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(scores)), scores, color=['green' if l==0 else 'red' for l in labels])
    plt.xlabel('Test Signal Index')
    plt.ylabel('Reconstruction Error (Anomaly Score)')
    plt.title('Anomaly Scores (Green=Normal, Red=Anomalous)')
    #plt.show()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "anomaly_scores.png"))
    plt.close()