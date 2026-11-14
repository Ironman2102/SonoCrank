"""compare.py file1.npy file2.npy ... -> prints how different the recordings are and saves compare.png
Example:  python compare.py test/off.npy test/on.npy test/coin.npy
Looks at the frequency content, not just the overall size: a running fan shows sharp peaks
(rotation and blade-pass tones) even when its total vibration is small."""
import sys, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SR = 16000
def spectrum(x):
    x = x.astype(float) - x.mean(); n = 4096; w = np.hanning(n)
    segs = [np.abs(np.fft.rfft(x[i:i + n] * w)) ** 2 for i in range(0, len(x) - n, n // 2)]
    return np.fft.rfftfreq(n, 1 / SR), np.mean(segs, axis=0)
files = sys.argv[1:]
if not files: sys.exit(__doc__)
res = {}
for f in files:
    x = np.load(f); fr, p = spectrum(x); res[f] = (fr, p)
    band = (fr > 20) & (fr < 4000)
    pk = np.argsort(p * band)[::-1]; peaks = []
    for i in pk:
        if all(abs(fr[i] - q) > 15 for q in peaks): peaks.append(fr[i])
        if len(peaks) == 4: break
    print(f"{os.path.basename(f):22s} RMS {x.astype(float).std():6.1f}   min/max {x.min():4d}/{x.max():4d}   "
          f"strongest tones (Hz): {', '.join('%.0f' % q for q in sorted(peaks))}")
if len(files) > 1:
    base = res[files[0]][1]
    for f in files[1:]:
        fr, p = res[f]; band = (fr > 20) & (fr < 4000)
        ratio = p[band] / (base[band] + 1e-9)
        print(f"{os.path.basename(f)} vs {os.path.basename(files[0])}: strongest tone is {10*np.log10(ratio.max()):.0f} dB above it "
              f"at {fr[band][ratio.argmax()]:.0f} Hz (10 dB or more = clearly visible to the model)")
plt.figure(figsize=(9, 4.5))
for f in files:
    fr, p = res[f]; plt.semilogy(fr, p + 1e-3, lw=1, label=os.path.basename(f))
plt.xlim(0, 2000); plt.xlabel("Frequency (Hz)"); plt.ylabel("Power"); plt.legend(); plt.grid(alpha=.3)
plt.title("Vibration spectrum"); plt.tight_layout(); plt.savefig("compare.png", dpi=120)
print("saved compare.png")
