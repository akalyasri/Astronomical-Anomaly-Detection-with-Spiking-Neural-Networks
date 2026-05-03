import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from preprocessing import preprocess, to_tensor
from encoding import rate_encode
from model import SimpleSNN
from astropy.io import fits

# auto create a results folder
# RESULTS_DIR = "results"
# os.makedirs(RESULTS_DIR, exist_ok=True)

threshold_val = 0.05  # change per experiment --> locked in the best encoding threshold now

RESULTS_DIR = f"results/threshold_{str(threshold_val).replace('.', '')}"
os.makedirs(RESULTS_DIR, exist_ok=True)

ANOMALY_TYPES = [
    "amplitude_burst",
    "frequency_shift",
    "phase_shift",
    "noise_burst",
    "dropout"
]

# synthetic data generation (1 vs. multiple)
def generate_signal(length=500, amplitude=1.0, frequency=1.0, phase=0.0,
                    anomaly=False, anomaly_type=None):
    t = np.linspace(0, 10, length)
    signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)

    if anomaly:
        if anomaly_type is None:
            anomaly_type = np.random.choice([
                "amplitude_burst",
                "frequency_shift",
                "phase_shift",
                "noise_burst",
                "dropout"
            ])

        start = np.random.randint(length // 4, length // 2)
        seg_len = np.random.randint(length // 12, length // 6)
        end = min(length, start + seg_len)

        if anomaly_type == "amplitude_burst":
            signal[start:end] += np.random.uniform(1.0, 2.0)

        elif anomaly_type == "frequency_shift":
            new_freq = np.random.uniform(1.5, 2.5)
            signal[start:end] = amplitude * np.sin(
                2 * np.pi * new_freq * t[start:end] + phase
            )

        elif anomaly_type == "phase_shift":
            signal[start:end] = amplitude * np.sin(
                2 * np.pi * frequency * t[start:end] + phase + np.pi / 2
            )

        elif anomaly_type == "noise_burst":
            signal[start:end] += np.random.normal(0, 0.8, end - start)

        elif anomaly_type == "dropout":
            signal[start:end] = 0.0

    return signal


def generate_signals(num_signals=50, anomaly=False, anomaly_type=None):
    signals = []
    for _ in range(num_signals):
        amp = np.random.uniform(0.8, 1.2)
        freq = np.random.uniform(0.8, 1.2)
        phase = np.random.uniform(0, np.pi / 4)

        signal = generate_signal(
            amplitude=amp,
            frequency=freq,
            phase=phase,
            anomaly=anomaly,
            anomaly_type=anomaly_type
        )
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
        spikes = rate_encode(sig_proc, threshold_val)
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
        spikes = rate_encode(sig_proc, threshold_val)
        tensor_sig = to_tensor(spikes)
        processed_signals.append(tensor_sig)

    x_batch = torch.stack(processed_signals, dim=1)
    with torch.no_grad():
        output = model(x_batch)
        mse_per_signal = ((output - x_batch)**2).mean(dim=0)
        scores = mse_per_signal.squeeze().numpy()

    return scores, x_batch, output


# evaluating
def evaluate_test_set(model, test_signals, labels, experiment_dir, experiment_name):
    os.makedirs(experiment_dir, exist_ok=True)

    scores, x_test, output_test = compute_anomaly_scores(model, test_signals, labels)

    num_test = labels.count(0)

    threshold = np.mean(scores[:num_test]) - 1 * np.std(scores[:num_test])
    print(f"\n===== {experiment_name} =====")
    print(f"Anomaly detection threshold (lower than normal) = {threshold:.4f}\n")

    preds = [1 if score < threshold else 0 for score in scores]

    for i, score in enumerate(scores):
        if preds[i] == 1:
            print(f"Signal {i:02d} flagged as ANOMALY (score = {score:.4f})")
        else:
            print(f"Signal {i:02d} considered NORMAL (score = {score:.4f})")

    tp = sum((p == 1 and y == 1) for p, y in zip(preds, labels))
    tn = sum((p == 0 and y == 0) for p, y in zip(preds, labels))
    fp = sum((p == 1 and y == 0) for p, y in zip(preds, labels))
    fn = sum((p == 0 and y == 1) for p, y in zip(preds, labels))

    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\nEvaluation Metrics:")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    with open(os.path.join(experiment_dir, "metrics.txt"), "w") as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Encoding threshold: {threshold_val}\n")
        f.write(f"Anomaly decision threshold: {threshold:.4f}\n")
        f.write(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

    np.save(os.path.join(experiment_dir, "scores.npy"), scores)
    np.save(os.path.join(experiment_dir, "preds.npy"), np.array(preds))
    np.save(os.path.join(experiment_dir, "labels.npy"), np.array(labels))

    idx_example = num_test

    plt.figure(figsize=(12, 4))
    plt.plot(x_test[:, idx_example, :].numpy(), label='Encoded Signal (Anomalous)')
    plt.plot(output_test[:, idx_example, :].numpy(), label='Reconstructed Signal')
    plt.title(f'SNN Reconstruction: {experiment_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, "reconstruction_example.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(scores)), scores, color=['green' if l == 0 else 'red' for l in labels])
    plt.xlabel('Test Signal Index')
    plt.ylabel('Reconstruction Error (Anomaly Score)')
    plt.title(f'Anomaly Scores: {experiment_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, "anomaly_scores.png"))
    plt.close()

    return {
        "experiment": experiment_name,
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def run_kepler_case_study(model, fits_path, experiment_dir):
    os.makedirs(experiment_dir, exist_ok=True)

    hdul = fits.open(fits_path)
    data = hdul[1].data

    time = data["TIME"]
    flux = data["PDCSAP_FLUX"]

    mask = ~np.isnan(time) & ~np.isnan(flux)
    time = time[mask]
    flux = flux[mask]

    # use first 500 points so it matches synthetic signal length
    time = time[:500]
    flux = flux[:500]

    # normalize Kepler flux first
    flux_norm = (flux - np.mean(flux)) / (np.std(flux) + 1e-8)

    # smooth slightly so encoding is less noisy
    window = 5
    flux_smooth = np.convolve(flux_norm, np.ones(window) / window, mode="same")

    # scale to [-1, 1] like synthetic signal preprocessing
    flux_scaled = flux_smooth / (np.max(np.abs(flux_smooth)) + 1e-8)

    # adaptive threshold based on Kepler signal variation
    kepler_encoding_threshold = 0.05 * np.std(flux_scaled)

    spikes = rate_encode(flux_scaled, kepler_encoding_threshold)
    tensor_sig = to_tensor(spikes)

    x_kepler = tensor_sig.unsqueeze(1)

    model.eval()
    with torch.no_grad():
        output = model(x_kepler)
        mse = ((output - x_kepler) ** 2).mean().item()

    # raw Kepler light curve
    plt.figure(figsize=(12, 4))
    plt.plot(time, flux)
    plt.xlabel("Time")
    plt.ylabel("PDCSAP Flux")
    plt.title("Raw Kepler Light Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, "kepler_raw_light_curve.png"))
    plt.close()


    # Kepler preprocessing outputs
    plt.figure(figsize=(12, 4))
    plt.plot(time, flux_scaled)
    plt.xlabel("Time")
    plt.ylabel("Normalized Flux")
    plt.title("Preprocessed Kepler Light Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, "kepler_preprocessed_light_curve.png"))
    plt.close()

    # encoded + reconstruction
    plt.figure(figsize=(12, 4))
    plt.plot(x_kepler[:, 0, :].numpy(), label="Encoded Kepler Signal")
    plt.plot(output[:, 0, :].numpy(), label="Reconstructed Signal")
    plt.xlabel("Time Step")
    plt.ylabel("Encoded Value")
    plt.title("SNN Reconstruction on Kepler Light Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, "kepler_reconstruction.png"))
    plt.close()

    with open(os.path.join(experiment_dir, "kepler_case_study.txt"), "w") as f:
        f.write("Kepler qualitative case study\n")
        f.write(f"File: {fits_path}\n")
        f.write(f"Synthetic encoding threshold: {threshold_val}\n")
        f.write(f"Kepler adaptive encoding threshold: {kepler_encoding_threshold:.6f}\n")
        f.write(f"Reconstruction MSE: {mse:.6f}\n")
        f.write("Note: No ground-truth anomaly labels were used. This is a qualitative transfer test.\n")

    print(f"\nKepler case study saved to: {experiment_dir}")
    print(f"Kepler reconstruction MSE: {mse:.6f}")


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

    # save training loss
    np.save(os.path.join(RESULTS_DIR, "losses.npy"), np.array(all_losses))

    # anomaly-type breakdown experiments
    num_test = 20
    all_results = []

    for anomaly_type in ANOMALY_TYPES:
        normal_test = generate_signals(num_test, anomaly=False)
        anomalous_test = generate_signals(num_test, anomaly=True, anomaly_type=anomaly_type)

        test_signals = normal_test + anomalous_test
        labels = [0] * num_test + [1] * num_test

        experiment_dir = os.path.join(RESULTS_DIR, f"anomaly_type_{anomaly_type}")

        result = evaluate_test_set(
            model=model,
            test_signals=test_signals,
            labels=labels,
            experiment_dir=experiment_dir,
            experiment_name=anomaly_type
        )

        all_results.append(result)

    # save summary table as CSV
    summary_path = os.path.join(RESULTS_DIR, "anomaly_type_summary.csv")

    with open(summary_path, "w") as f:
        f.write("experiment,threshold,tp,tn,fp,fn,accuracy,precision,recall,f1\n")
        for r in all_results:
            f.write(
                f"{r['experiment']},{r['threshold']:.4f},"
                f"{r['tp']},{r['tn']},{r['fp']},{r['fn']},"
                f"{r['accuracy']:.4f},{r['precision']:.4f},"
                f"{r['recall']:.4f},{r['f1']:.4f}\n"
            )

    print(f"\nSaved anomaly-type summary to: {summary_path}")

    kepler_path = "../data/kplr008462852-2009131105131_llc.fits"
    kepler_dir = os.path.join(RESULTS_DIR, "kepler_case_study")

    run_kepler_case_study(
        model=model,
        fits_path=kepler_path,
        experiment_dir=kepler_dir
    )