"""anomaly.py - shared code for learning 'healthy' only (no fault recordings needed).
Two detectors, both trained on healthy windows only:
  1. z-score baseline: the same maths the ESP32 detector runs (per-coefficient mean/std of MFCC)
  2. autoencoder: a small neural network that learns to rebuild healthy MFCC windows;
     anything it rebuilds badly is unlike healthy, so its error is the anomaly score.
Simulated faults are added to real healthy recordings to measure how well each detector catches them."""
import os, numpy as np, features as F

HOP = 8000                         # a new 1 s window every 0.5 s

def windows(x, hop=HOP):
    return np.array([F.mfcc_window(x[i:i + F.WIN_SAMPLES]) for i in range(0, len(x) - F.WIN_SAMPLES + 1, hop)])

# ---------- simulated faults, added to real healthy signal ----------
def inject(x, kind, severity=1.0, seed=0):
    """x: raw ADC counts. Returns a copy with a simulated fault added. severity scales the fault size
    relative to the healthy vibration's own RMS, so it adapts to how strong your engine signal is."""
    r = np.random.default_rng(seed); y = x.astype(np.float64); s = y.std() + 1e-9; n = len(y); t = np.arange(n) / F.SR
    if kind == "rattle":           # loose part: random short metallic clicks, about 12 per second
        k = np.arange(int(0.006 * F.SR)) / F.SR
        for p in r.integers(0, n - len(k), int(12 * n / F.SR)):
            f = r.uniform(1500, 3500); y[p:p + len(k)] += severity * 2.0 * s * r.uniform(.5, 1.5) * np.exp(-k / 0.0015) * np.sin(2 * np.pi * f * k)
    elif kind == "whine":          # worn bearing / belt: a steady high tone
        y += severity * 0.4 * s * np.sin(2 * np.pi * 1850 * t)
    elif kind == "knock":          # heavy periodic thump (loose mount, misfire-like): about 6 per second
        k = np.arange(int(0.04 * F.SR)) / F.SR
        for p in np.arange(0, n - len(k), int(F.SR / 6.3)):
            y[p:p + len(k)] += severity * 5.0 * s * np.exp(-k / 0.012) * np.sin(2 * np.pi * 85 * k)
    else: raise ValueError(kind)
    return np.clip(np.round(y), 0, 4095).astype(np.uint16)
FAULTS = ["rattle", "whine", "knock"]

# ---------- detector 1: z-score baseline (same as the ESP32) ----------
class ZScore:
    def fit(self, W):
        v = W.mean(1); self.mu = v.mean(0); self.sd = np.maximum(v.std(0), 0.1); return self
    def score(self, W):
        return np.sqrt((((W.mean(1) - self.mu) / self.sd) ** 2).mean(1))

# ---------- detector 2: autoencoder ----------
# Input = 26 window statistics: mean and std over the 30 frames of each of the 13 MFCCs.
# (The first version fed all 390 per-frame values. Frame-to-frame noise is 3-6x larger than the
#  noise of the window mean, so the AE's alert threshold had to be high and it missed real imbalance.
#  See AE_review.md. The 26-input AE also fits on the ESP32: about 1,100 parameters.)
def ae_features(W):
    return np.concatenate([W.mean(1), W.std(1)], 1)

def build_ae(d=2 * F.N_MFCC, hidden=16, latent=6):
    import tensorflow as tf
    inp = tf.keras.Input(shape=(d,))
    h = tf.keras.layers.Dense(hidden, activation="relu")(inp)
    h = tf.keras.layers.Dense(latent, activation="relu")(h)
    h = tf.keras.layers.Dense(hidden, activation="relu")(h)
    out = tf.keras.layers.Dense(d)(h)
    return tf.keras.Model(inp, out)

class AE:
    def fit(self, W, epochs=60, seed=0):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        X = ae_features(W); self.mu = X.mean(0); self.sd = X.std(0) + 1e-3     # per-feature, healthy training data only
        Xn = (X - self.mu) / self.sd
        self.m = build_ae(X.shape[1]); self.m.compile("adam", "mse")
        self.m.fit(Xn, Xn, epochs=epochs, batch_size=32, verbose=0, shuffle=True); return self
    def norm(self, W): return ((ae_features(W) - self.mu) / self.sd).astype(np.float32)
    def score(self, W):
        Xn = self.norm(W)
        return np.sqrt(((self.m.predict(Xn, verbose=0) - Xn) ** 2).mean(1))

def threshold(train_scores):
    """Alert level: comfortably above anything seen while healthy (same rule as the ESP32: 1.5 x max)."""
    return 1.5 * float(np.max(train_scores))

def alerts(scores, thr, need=4, of=6):
    """ESP32-style voting: ALERT when at least 4 of the last 6 windows are above the threshold."""
    above = scores > thr; return np.array([above[max(0, i - of + 1):i + 1].sum() >= need for i in range(len(above))])
