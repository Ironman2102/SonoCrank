"""train_healthy.py - train and test using HEALTHY recordings only.

  Recordings:  data/healthy/s1_idle.npy, s2_idle.npy, ... (one file per session, engine restarted in between)
  Run:         python train_healthy.py

How it is tested, honestly, without any real fault recordings:
  * each session in turn is held out; the detectors learn from the other sessions only
  * false alarms: how often the held-out HEALTHY session triggers an alert
  * detection: simulated faults (rattle, whine, knock) are added to the held-out healthy signal,
    and we measure how much of the faulty time is flagged, and how fast the first alert comes
Writes results/healthy_summary.txt, results/healthy_scores.png, results/healthy_results.json,
and the final model files ae_int8.tflite + ae_norm.npz (used by live_anomaly.py)."""
import os, glob, json, argparse, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2"); warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import features as F, anomaly as A

ap = argparse.ArgumentParser(); ap.add_argument("--folder", default="data/healthy"); ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--severity", type=float, default=1.0, help="size of the simulated faults (1.0 = default)")
a = ap.parse_args()
files = sorted(glob.glob(os.path.join(a.folder, "*.npy")))
if len(files) < 3: raise SystemExit(f"Need at least 3 session files in {a.folder}, found {len(files)}")
os.makedirs("results", exist_ok=True)
raw = {os.path.basename(f).split("_")[0]: np.load(f) for f in files}
print(f"{len(raw)} healthy sessions:", ", ".join(f"{k} ({len(v)/F.SR:.0f} s)" for k, v in raw.items()))
W = {k: A.windows(x) for k, x in raw.items()}
root = os.path.dirname(os.path.normpath(a.folder)) or "."
real = {}                                   # real fault recordings: {class: {session: windows}}
for d in sorted(os.listdir(root)):
    p = os.path.join(root, d)
    if os.path.normpath(p) == os.path.normpath(a.folder) or not os.path.isdir(p): continue
    fs = sorted(glob.glob(os.path.join(p, "*.npy")))
    if fs: real[d] = {os.path.basename(f).split("_")[0]: A.windows(np.load(f)) for f in fs}
if real: print("real fault recordings (used only for testing):", ", ".join(f"{c} ({len(v)})" for c, v in real.items()))

SEV = [0.5, 1.0, 2.0]                       # small, medium, large simulated faults (x --severity)
res = {"sessions": {}, "faults": A.FAULTS, "severity_levels": [a.severity * v for v in SEV]}
for held in raw:
    train = np.concatenate([W[k] for k in raw if k != held])
    dets = {"zscore": A.ZScore().fit(train), "autoencoder": A.AE().fit(train, a.epochs)}
    r = {}
    for name, d in dets.items():
        thr = A.threshold(d.score(train)); h = d.score(W[held])
        r[name] = {"thr": thr, "healthy_alert_pct": 100 * A.alerts(h, thr).mean()}
        for fk in A.FAULTS:
            for sv in SEV:
                s = d.score(A.windows(A.inject(raw[held], fk, a.severity * sv, seed=int(held[1:] or 0) if held[1:].isdigit() else 0)))
                al = A.alerts(s, thr)
                r[name][f"{fk}@{sv}"] = {"detected_pct": 100 * al.mean(), "first_alert_s": (1.0 + np.argmax(al) * A.HOP / F.SR) if al.any() else None}
        for c, sess in real.items():
            if held in sess: r[name][f"real:{c}"] = {"detected_pct": 100 * A.alerts(d.score(sess[held]), thr).mean(), "first_alert_s": None}
        print(f"  held out {held:4s} {name:11s} false alerts {r[name]['healthy_alert_pct']:5.1f}%   medium faults caught: "
              + "  ".join(f"{fk} {r[name][fk + '@1.0']['detected_pct']:5.1f}%" for fk in A.FAULTS))
    res["sessions"][held] = r

def avg(name, key, sub):
    v = [res["sessions"][s][name][key][sub] if sub else res["sessions"][s][name][key] for s in raw]
    v = [x for x in v if x is not None]; return float(np.mean(v)) if v else float("nan")
L = [f"Healthy-only training and testing: {len(raw)} sessions, {sum(len(v) for v in raw.values())/F.SR:.0f} s of real healthy recordings.",
     "Each session is held out in turn; the detector learns from the other sessions only.",
     "Faults are SIMULATED by adding them to the held-out healthy recording (small / medium / large).", "",
     "False alerts on held-out healthy data (% of time with the alert on):"]
for name in ["zscore", "autoencoder"]: L.append(f"  {name:12s} {avg(name, 'healthy_alert_pct', None):5.1f}%")
L += ["", "Faults caught (% of the faulty time with the alert on; alert needs 4 of the last 6 windows):",
      f"  {'':12s}" + "".join(f"{fk:>21s}" for fk in A.FAULTS), f"  {'':12s}" + "".join(f"{'small  med  large':>21s}" for _ in A.FAULTS)]
for name in ["zscore", "autoencoder"]:
    L.append(f"  {name:12s}" + "".join("    " + " ".join(f"{avg(name, f'{fk}@{sv}', 'detected_pct'):5.0f}" for sv in SEV) for fk in A.FAULTS))
if real:
    L += ["", "REAL recorded faults, never used in training (% of time with the alert on, same held-out session):"]
    for name in ["zscore", "autoencoder"]:
        def avg_real(c):
            v = [res["sessions"][s_][name]["real:" + c]["detected_pct"] for s_ in raw if ("real:" + c) in res["sessions"][s_][name]]
            return f"{c} {np.mean(v):5.1f}% ({len(v)} sessions)" if v else f"{c} n/a (no matching session names)"
        L.append(f"  {name:12s}" + "   ".join(avg_real(c) for c in real))
L += ["", "Time to first alert, medium faults (s after the window starts filling):"]
for name in ["zscore", "autoencoder"]:
    L.append(f"  {name:12s}" + "  ".join(f"{fk} {avg(name, fk + '@1.0', 'first_alert_s'):4.1f}" for fk in A.FAULTS))
L += ["", "zscore = the method running on the ESP32 detector (detector.ino). autoencoder = 26-16-6-16-26 on window mean+std of the MFCCs (ae_int8.tflite)."]
open("results/healthy_summary.txt", "w").write("\n".join(L) + "\n"); print("\n" + "\n".join(L))
json.dump(res, open("results/healthy_results.json", "w"), indent=1, default=float)

# final models on all sessions + figure for the report
allW = np.concatenate(list(W.values()))
z = A.ZScore().fit(allW); ae = A.AE().fit(allW, a.epochs)
thr_z, thr_ae = A.threshold(z.score(allW)), A.threshold(ae.score(allW))
first = list(raw)[0]; x = raw[first]; half = len(x) // 2
fault_name = "simulated rattle"
if real:
    c0 = list(real)[0]; f0 = sorted(glob.glob(os.path.join(root, c0, "*.npy")))[0]
    demo = np.concatenate([x[:half], np.load(f0)[:len(x) - half]]); fault_name = f"real {c0} recording"
else:
    demo = np.concatenate([x[:half], A.inject(x[half:], "rattle", a.severity, seed=1)])      # healthy, then rattle starts
Wd = A.windows(demo); t = 1.0 + np.arange(len(Wd)) * A.HOP / F.SR
fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
for axi, (nm, d, thr) in zip(ax, [("z-score (ESP32 method)", z, thr_z), ("autoencoder", ae, thr_ae)]):
    s = d.score(Wd); axi.plot(t, s, lw=1.4, color="#1f6fe0"); axi.axhline(thr, color="#d1262b", ls="--", lw=1, label="alert threshold")
    axi.axvspan(half / F.SR, len(x) / F.SR, color="#f2cf1d", alpha=.25, label=fault_name); axi.set_ylabel("anomaly score"); axi.set_title(nm, fontsize=10); axi.legend(fontsize=8, loc="upper left")
ax[1].set_xlabel("time (s)"); plt.tight_layout(); plt.savefig("results/healthy_scores.png", dpi=150); plt.close()

import train as T
Xn = ae.norm(allW)
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    T.to_int8_tflite(ae.m, Xn, "ae_int8.tflite")
np.savez("ae_norm.npz", mu=ae.mu, sd=ae.sd, thr=thr_ae, z_mu=z.mu, z_sd=z.sd, z_thr=thr_z)
print(f"\nsaved results/healthy_summary.txt, results/healthy_scores.png, ae_int8.tflite ({os.path.getsize('ae_int8.tflite')/1024:.1f} KB), ae_norm.npz")
