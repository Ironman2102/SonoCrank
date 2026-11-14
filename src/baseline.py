import numpy as np
class Baseline:
    """Per-bin running mean/variance (Welford) of the window-mean MFCC vector; score = RMS z-score."""
    def __init__(self, n_bins=3, dim=13):
        self.n = np.zeros(n_bins); self.mu = np.zeros((n_bins, dim)); self.m2 = np.zeros((n_bins, dim))
    def update(self, b, v):
        self.n[b] += 1; d = v - self.mu[b]; self.mu[b] += d / self.n[b]; self.m2[b] += d * (v - self.mu[b])
    def score(self, b, v):
        sd = np.sqrt(self.m2[b] / max(self.n[b] - 1, 1)) + 1e-3
        return float(np.sqrt(np.mean(((v - self.mu[b]) / sd) ** 2)))
