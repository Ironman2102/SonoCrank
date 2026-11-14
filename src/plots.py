"""plots.py - report figures from your recordings.
  python plots.py      ->  results/signals.png (waveform + spectrogram per condition)
                           results/spectra.png (average spectrum per condition)
                           results/mfcc.png    (one 1-second MFCC window per condition)"""
import glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import features as F

os.makedirs("results", exist_ok=True)
classes = sorted(d for d in os.listdir("data") if glob.glob(os.path.join("data", d, "*.npy")))
first = {c: sorted(glob.glob(os.path.join("data", c, "*.npy")))[0] for c in classes}
volts = lambda x: (x.astype(float) - x.mean()) * 3.3 / 4095

fig, axes = plt.subplots(2, len(classes), figsize=(4.2 * len(classes), 6), dpi=150, squeeze=False)
for k, c in enumerate(classes):
    x = np.load(first[c])[: 2 * F.SR]
    t = np.arange(len(x)) / F.SR
    axes[0, k].plot(t, volts(x), lw=0.4); axes[0, k].set_title(c); axes[0, k].set_xlabel("Time (s)"); axes[0, k].set_ylabel("Signal (V, AC)")
    axes[1, k].specgram(volts(x), NFFT=1024, Fs=F.SR, noverlap=512, cmap="magma")
    axes[1, k].set_ylim(0, 4000); axes[1, k].set_xlabel("Time (s)"); axes[1, k].set_ylabel("Frequency (Hz)")
fig.tight_layout(); fig.savefig("results/signals.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
for c in classes:
    x = volts(np.load(first[c])); segs = [x[i:i + 4096] * np.hanning(4096) for i in range(0, len(x) - 4096, 2048)]
    p = np.mean([np.abs(np.fft.rfft(s)) ** 2 for s in segs], axis=0); f = np.fft.rfftfreq(4096, 1 / F.SR)
    ax.semilogy(f, p, lw=0.8, label=c)
ax.set_xlim(0, 4000); ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Power (log)"); ax.legend(); ax.set_title("Average vibration spectrum")
fig.tight_layout(); fig.savefig("results/spectra.png"); plt.close(fig)

fig, axes = plt.subplots(1, len(classes), figsize=(3.4 * len(classes), 3), dpi=150, squeeze=False)
for k, c in enumerate(classes):
    m = F.mfcc_window(np.load(first[c]))
    im = axes[0, k].imshow(m.T, aspect="auto", origin="lower", cmap="viridis")
    axes[0, k].set_title(c); axes[0, k].set_xlabel("Frame (32 ms)"); axes[0, k].set_ylabel("MFCC index")
fig.tight_layout(); fig.savefig("results/mfcc.png"); plt.close(fig)
print("Saved results/signals.png, results/spectra.png, results/mfcc.png")
