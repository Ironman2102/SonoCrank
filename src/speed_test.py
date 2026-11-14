"""speed_test.py - does the detector still work when the engine speed changes?
Healthy recordings are replayed slower and faster (every frequency x k, vibration x k^2) to simulate engine speeds.
Leave-one-session-out: train on 5 healthy sessions, test on the 6th at speeds NOT used in training.
  python speed_test.py   -> results/speed_summary.txt"""
import os; os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np, anomaly as A
from export_ae import resample
S = [f"s{i}" for i in range(1, 7)]; TRAIN_K = [0.7, 0.85, 1.0, 1.2, 1.4]; TEST_H = [0.75, 0.95, 1.1, 1.3]; TEST_F = [0.75, 1.1, 1.3]
raw = {c: {s: np.load(f"data/{c}/{s}_speed1.npy") for s in S} for c in ["healthy", "imbalance", "rubbing"]}
cache = {}
def win(c, s, k):
    if (c, s, k) not in cache: cache[(c, s, k)] = A.windows(resample(raw[c][s], k) if k != 1.0 else raw[c][s])
    return cache[(c, s, k)]
def zs(tr):
    v = tr.mean(1); mu, sd = v.mean(0), np.maximum(v.std(0), 0.1)
    f = lambda W: np.sqrt((((W.mean(1) - mu) / sd) ** 2).mean(1)); thr = A.threshold(f(tr)); return lambda W: f(W) / thr
R = {}
for held in S:
    tr_all = np.concatenate([win("healthy", s, k) for s in S if s != held for k in TRAIN_K])
    tr_one = np.concatenate([win("healthy", s, 1.0) for s in S if s != held])
    ae = A.AE().fit(tr_all, 60); thr = A.threshold(ae.score(tr_all))
    D = {"z-score learned at one speed (ESP32 z-score mode)": zs(tr_one), "z-score with all speeds in one baseline": zs(tr_all),
         "speed-aware autoencoder (ESP32 autoencoder mode)": lambda W: ae.score(W) / thr}
    for n, f in D.items():
        r = R.setdefault(n, [])
        r.append([np.mean([A.alerts(f(win("healthy", held, k)), 1.0).mean() for k in TEST_H]) * 100,
                  np.mean([A.alerts(f(win("imbalance", held, k)), 1.0).mean() for k in TEST_F]) * 100,
                  np.mean([A.alerts(f(win("rubbing", held, k)), 1.0).mean() for k in TEST_F]) * 100])
L = ["Changing engine speed (simulated by replaying recordings slower/faster), leave-one-session-out, healthy-only training.",
     f"Training speeds x{TRAIN_K}; healthy tested at unseen speeds x{TEST_H}; real faults tested at x{TEST_F}.", "",
     f"{'detector':52s} {'false alerts':>12s} {'imbalance':>10s} {'rubbing':>9s}"]
for n, r in R.items():
    m = np.mean(r, 0); L.append(f"{n:52s} {m[0]:11.1f}% {m[1]:9.1f}% {m[2]:8.1f}%")
L += ["", "% = share of time with the alert on (alert needs 4 of the last 6 windows; 97.5% is the most possible)."]
os.makedirs("results", exist_ok=True); open("results/speed_summary.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
