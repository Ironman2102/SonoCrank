# ae_review_exp.py - autoencoder review experiments (see AE_review.md). Run ae_review_prep.py first.
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pickle, json, anomaly as A, tensorflow as tf
W, SIM, RMS, RMSSIM = pickle.load(open("win.pkl", "rb")); S = [f"s{i}" for i in range(1, 7)]; SEV = [0.5, 1.0, 2.0]

class AEx:                              # one autoencoder, several ways of scoring it
    def fit(self, X, epochs=60, seed=0, latent=12, hidden=48):
        tf.keras.utils.set_random_seed(seed); self.shape = X.shape[1:]
        X = X.reshape(len(X), -1); self.mu = X.mean(0); self.sd = X.std(0) + 1e-3; Xn = (X - self.mu) / self.sd
        inp = tf.keras.Input(shape=(Xn.shape[1],)); h = tf.keras.layers.Dense(hidden, activation="relu")(inp)
        z = tf.keras.layers.Dense(latent, activation="relu")(h); h2 = tf.keras.layers.Dense(hidden, activation="relu")(z)
        self.m = tf.keras.Model(inp, tf.keras.layers.Dense(Xn.shape[1])(h2)); self.enc = tf.keras.Model(inp, z)
        self.m.compile("adam", "mse"); self.m.fit(Xn, Xn, epochs=epochs, batch_size=32, verbose=0, shuffle=True)
        C = self.enc.predict(Xn, verbose=0); self.cmu = C.mean(0); self.cinv = np.linalg.pinv(np.cov(C.T) + 1e-3 * np.eye(C.shape[1])); return self
    def parts(self, X):
        Xn = (X.reshape(len(X), -1) - self.mu) / self.sd; R = self.m.predict(Xn, verbose=0) - Xn
        return Xn, R
    def s_mean(self, X): _, R = self.parts(X); return np.sqrt((R ** 2).mean(1))                          # current method
    def s_framemean(self, X): _, R = self.parts(X); r = R.reshape(len(X), *self.shape).mean(1); return np.sqrt((r ** 2).mean(1))   # residual averaged over frames
    def s_max(self, X): _, R = self.parts(X); r = R.reshape(len(X), *self.shape); return np.sqrt((r ** 2).mean(2)).max(1)        # worst frame
    def s_coefw(self, X): _, R = self.parts(X); r = R.reshape(len(X), *self.shape); return np.sqrt((r ** 2).mean(1)).max(1)      # worst coefficient
    def s_latent(self, X):
        Xn = (X.reshape(len(X), -1) - self.mu) / self.sd; d = self.enc.predict(Xn, verbose=0) - self.cmu
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self.cinv, d))

def feat26(Wx): return np.concatenate([Wx.mean(1), Wx.std(1)], 1)
def zscore(tr, Wx): v = tr.mean(1); mu, sd = v.mean(0), np.maximum(v.std(0), 0.1); return np.sqrt((((Wx.mean(1) - mu) / sd) ** 2).mean(1))
def zframemax(tr, Wx): f = tr.reshape(-1, 13); mu, sd = f.mean(0), f.std(0) + 1e-3; return np.sqrt((((Wx - mu) / sd) ** 2).mean(2)).max(1)

def evalfold(held):
    tr = np.concatenate([W["healthy"][s] for s in S if s != held])
    tests = {"healthy": W["healthy"][held], "imbalance": W["imbalance"][held], "rubbing": W["rubbing"][held]}
    tests.update({f"{f}@{v}": SIM[(held, f, v)] for f in A.FAULTS for v in SEV})
    rtr = np.concatenate([RMS["healthy"][s] for s in S if s != held])
    rtests = {"healthy": RMS["healthy"][held], "imbalance": RMS["imbalance"][held], "rubbing": RMS["rubbing"][held]}
    rtests.update({f"{f}@{v}": RMSSIM[(held, f, v)] for f in A.FAULTS for v in SEV})
    ae = AEx().fit(tr); ae26 = AEx().fit(feat26(tr), latent=6, hidden=16)
    det = {"RMS only (loudness)": (lambda X, k=None: X, rtr, rtests),
           "z-score (ESP32)": (lambda X: zscore(tr, X), tr, tests),
           "z-score, worst frame": (lambda X: zframemax(tr, X), tr, tests),
           "AE current (mean error)": (ae.s_mean, tr, tests),
           "AE frame-averaged residual": (ae.s_framemean, tr, tests),
           "AE latent Mahalanobis": (ae.s_latent, tr, tests),
           "AE max-over-frames": (ae.s_max, tr, tests),
           "AE worst coefficient": (ae.s_coefw, tr, tests),
           "AE-26 (mean+std, 26-16-6-16-26)": (lambda X: ae26.s_mean(feat26(X)), tr, tests)}
    out, flags = {}, {}
    for name, (fn, trd, td) in det.items():
        thr = A.threshold(fn(trd)); out[name] = {}; flags[name] = {}
        for k, X in td.items():
            al = A.alerts(fn(X), thr); flags[name][k] = al
            out[name][k] = (100 * al.mean(), (1.0 + np.argmax(al) * 0.5) if al.any() else None)
    for a, b in (("z-score (ESP32)", "AE current (mean error)"), ("z-score (ESP32)", "AE frame-averaged residual")):
        nm = f"Hybrid: {a.split(' ')[0]} OR {b}"; out[nm] = {k: (100 * (flags[a][k] | flags[b][k]).mean(),
              (1.0 + np.argmax(flags[a][k] | flags[b][k]) * 0.5) if (flags[a][k] | flags[b][k]).any() else None) for k in flags[a]}
    return out

res = {s: evalfold(s) for s in S}
names = list(res["s1"]); keys = list(res["s1"][names[0]])
def m(n, k, i): v = [res[s][n][k][i] for s in S if res[s][n][k][i] is not None]; return np.mean(v) if v else float("nan")
print(f"{'detector':34s} {'falseA':>6s} {'IMB':>5s} {'RUB':>5s} {'t_IMB':>5s} {'t_RUB':>5s} | " + " ".join(f"{f[:5]}{v:<4}" for f in A.FAULTS for v in SEV))
for n in names:
    print(f"{n:34s} {m(n,'healthy',0):6.1f} {m(n,'imbalance',0):5.1f} {m(n,'rubbing',0):5.1f} {m(n,'imbalance',1):5.1f} {m(n,'rubbing',1):5.1f} | "
          + " ".join(f"{m(n, f'{f}@{v}', 0):9.0f}" for f in A.FAULTS for v in SEV))
print("\nper-session real imbalance % (AE current / AE frame-avg / z):")
for s in S: print(s, [round(res[s][n]["imbalance"][0], 1) for n in ("AE current (mean error)", "AE frame-averaged residual", "z-score (ESP32)")])
json.dump({s: {n: {k: list(v) for k, v in res[s][n].items()} for n in res[s]} for s in S}, open("ae_review_results.json", "w"), default=float)
