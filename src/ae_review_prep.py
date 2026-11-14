# Precompute MFCC windows for all real recordings and all simulated faults (held-out healthy + injection)
import numpy as np, anomaly as A, pickle
S = [f"s{i}" for i in range(1, 7)]; SEV = [0.5, 1.0, 2.0]
raw = {c: {s: np.load(f"data/{c}/{s}_speed1.npy") for s in S} for c in ["healthy", "imbalance", "rubbing"]}
W = {c: {s: A.windows(raw[c][s]) for s in S} for c in raw}
SIM = {(s, f, v): A.windows(A.inject(raw["healthy"][s], f, v, seed=int(s[1:]))) for s in S for f in A.FAULTS for v in SEV}
RMS = {c: {s: np.array([raw[c][s][i:i + 16000].astype(float).std() for i in range(0, len(raw[c][s]) - 16000 + 1, 8000)]) for s in S} for c in raw}
RMSSIM = {(s, f, v): np.array([x[i:i + 16000].astype(float).std() for i in range(0, len(x) - 16000 + 1, 8000)])
          for s in S for f in A.FAULTS for v in SEV for x in [A.inject(raw["healthy"][s], f, v, seed=int(s[1:]))]}
pickle.dump((W, SIM, RMS, RMSSIM), open("win.pkl", "wb")); print("ok", W["healthy"]["s1"].shape)
