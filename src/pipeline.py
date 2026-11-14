"""pipeline.py - train and test the CNN on your recordings.

Recordings go in  data/<condition>/<session>_<anything>.npy   e.g.  data/imbalance/s2_speed1.npy
Every folder inside data/ that holds .npy files becomes one class.

  python pipeline.py              test on each session in turn (the model never sees that session
                                  while training), then train the final model on everything
  python pipeline.py --epochs 60

Writes results/ (summary.txt, results.json, confusion matrices as PNG) and the deployable model:
model_int8.tflite + norm.npz (used by live_classify.py) + norm.h (for the ESP32)."""
import argparse, contextlib, glob, io, json, os, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import features as F, train as T

def load_files():
    classes = sorted(d for d in os.listdir("data") if glob.glob(os.path.join("data", d, "*.npy")))
    files = []
    for ci, c in enumerate(classes):
        for p in sorted(glob.glob(os.path.join("data", c, "*.npy"))):
            files.append((ci, os.path.basename(p).split("_")[0], p))
    return classes, files

def windows(path, hop=8000):
    x = np.load(path)
    return np.array([F.mfcc_window(x[i:i + F.WIN_SAMPLES]) for i in range(0, len(x) - F.WIN_SAMPLES + 1, hop)])

def fit(X, y, n_cls, epochs, seed):
    tf.keras.utils.set_random_seed(seed)
    mu, sd = X.mean((0, 1)), X.std((0, 1)) + 1e-6
    m = T.build(n_cls)
    m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    m.fit(((X - mu) / sd)[..., None], y, epochs=epochs, batch_size=32, verbose=0, shuffle=True)
    return m, mu, sd

def int8_predict(path, Xn):
    it = tf.lite.Interpreter(model_path=path); it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]; s, z = i["quantization"]
    out = []
    for x in Xn:
        it.set_tensor(i["index"], np.clip(np.round(x[None] / s + z), -128, 127).astype(np.int8)); it.invoke()
        out.append(int(it.get_tensor(o["index"]).argmax()))
    return np.array(out), (s, z)

def confusion(y, p, n):
    cm = np.zeros((n, n), int)
    for a, b in zip(y, p): cm[a, b] += 1
    return cm

def plot_cm(cm, classes, title, path):
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(classes), 1.2 + 1.2 * len(classes)), dpi=160)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for r in range(len(classes)):
        for c in range(len(classes)):
            ax.text(c, r, f"{cm[r, c]}\n{100 * norm[r, c]:.0f}%", ha="center", va="center",
                    color="white" if norm[r, c] > 0.55 else "black", fontsize=9)
    ax.set_xticks(range(len(classes)), classes, rotation=20); ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted condition"); ax.set_ylabel("True condition"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=40); a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    classes, files = load_files(); n = len(classes)
    sessions = sorted({s for _, s, _ in files})
    print(f"conditions: {classes}\nsessions:   {sessions}\nfiles:      {len(files)}")
    if n < 2: raise SystemExit("Need recordings for at least 2 conditions in data/")
    t0 = time.time(); feats = {p: windows(p) for _, _, p in files}
    print(f"features:   {sum(len(v) for v in feats.values())} one-second windows ({time.time() - t0:.0f} s)")

    folds, total_cm, total_cm8 = [], np.zeros((n, n), int), np.zeros((n, n), int)
    if len(sessions) >= 2:
        for s in sessions:
            tr = [(ci, p) for ci, ss, p in files if ss != s]; te = [(ci, p) for ci, ss, p in files if ss == s]
            if len({ci for ci, _ in te}) < 1 or len({ci for ci, _ in tr}) < n: continue
            Xtr = np.concatenate([feats[p] for _, p in tr]); ytr = np.concatenate([[ci] * len(feats[p]) for ci, p in tr])
            Xte = np.concatenate([feats[p] for _, p in te]); yte = np.concatenate([[ci] * len(feats[p]) for ci, p in te])
            m, mu, sd = fit(Xtr, ytr, n, a.epochs, seed=1)
            Xte_n = ((Xte - mu) / sd)[..., None].astype("float32")
            pf = m.predict(Xte_n, verbose=0).argmax(1)
            with contextlib.redirect_stdout(io.StringIO()):
                T.to_int8_tflite(m, ((Xtr - mu) / sd)[..., None].astype("float32"), "results/_fold.tflite")
            p8, _ = int8_predict("results/_fold.tflite", Xte_n)
            cm, cm8 = confusion(yte, pf, n), confusion(yte, p8, n)
            total_cm += cm; total_cm8 += cm8
            fold = {"held_out": s, "windows": int(len(yte)), "acc_float": float((pf == yte).mean()), "acc_int8": float((p8 == yte).mean()),
                    "recall": {classes[c]: float(cm8[c, c] / cm8[c].sum()) for c in range(n) if cm8[c].sum()}}
            folds.append(fold)
            print(f"  held out {s}: {len(yte):4d} windows  float {fold['acc_float']:.3f}  int8 {fold['acc_int8']:.3f}")
        if os.path.exists("results/_fold.tflite"): os.remove("results/_fold.tflite")
        plot_cm(total_cm8, classes, "Held-out sessions, int8 model (all folds)", "results/confusion_heldout.png")
    else:
        print("Only one session found: record at least 2 sessions (s1_..., s2_...) for an honest test.")

    # final model on everything
    X = np.concatenate([feats[p] for _, _, p in files]); y = np.concatenate([[ci] * len(feats[p]) for ci, _, p in files])
    m, mu, sd = fit(X, y, n, a.epochs, seed=1)
    Xn = ((X - mu) / sd)[..., None].astype("float32")
    with contextlib.redirect_stdout(io.StringIO()):
        T.to_int8_tflite(m, Xn, "model_int8.tflite")
    _, (s8, z8) = int8_predict("model_int8.tflite", Xn[:1])
    np.savez("norm.npz", mu=mu, sd=sd, classes=np.array(classes))
    with open("norm.h", "w") as f:
        f.write("#pragma once\n")
        f.write("static const float MFCC_MEAN[13]={%s};\n" % ",".join("%.6g" % v for v in mu))
        f.write("static const float MFCC_STD[13]={%s};\n" % ",".join("%.6g" % v for v in sd))
        f.write("static const float IN_SCALE=%.9g; static const int IN_ZERO=%d;\n" % (s8, z8))
        f.write("static const char* CLASSES[%d]={%s};\n" % (n, ",".join('"%s"' % c for c in classes)))

    size = os.path.getsize("model_int8.tflite")
    acc = [f["acc_int8"] for f in folds]; accf = [f["acc_float"] for f in folds]
    res = {"classes": classes, "sessions": sessions, "windows_per_class": {c: int((y == i).sum()) for i, c in enumerate(classes)},
           "folds": folds, "mean_acc_int8": float(np.mean(acc)) if acc else None, "std_acc_int8": float(np.std(acc)) if acc else None,
           "mean_acc_float": float(np.mean(accf)) if accf else None, "parameters": int(m.count_params()), "model_bytes": size,
           "confusion_heldout_int8": total_cm8.tolist()}
    json.dump(res, open("results/results.json", "w"), indent=2)
    lines = [f"Conditions: {', '.join(classes)}", f"Sessions: {', '.join(sessions)}",
             "Windows per condition: " + ", ".join(f"{c} {k}" for c, k in res["windows_per_class"].items()),
             f"Model: {res['parameters']} parameters, {size / 1024:.1f} KB as int8 TFLite", ""]
    if folds:
        lines.append("Held-out session | windows | float acc | int8 acc | " + " | ".join(f"{c} recall" for c in classes))
        for f_ in folds:
            lines.append(f"{f_['held_out']} | {f_['windows']} | {100 * f_['acc_float']:.1f}% | {100 * f_['acc_int8']:.1f}% | " +
                         " | ".join(f"{100 * f_['recall'].get(c, float('nan')):.1f}%" for c in classes))
        lines.append(f"MEAN | | {100 * res['mean_acc_float']:.1f}% | {100 * res['mean_acc_int8']:.1f}% ± {100 * res['std_acc_int8']:.1f}% |")
        lines += ["", "Confusion matrix, all held-out windows, int8 (rows = true, columns = predicted):",
                  "            " + " ".join(f"{c[:10]:>10}" for c in classes)]
        for i, c in enumerate(classes): lines.append(f"{c[:10]:>10}  " + " ".join(f"{v:>10}" for v in total_cm8[i]))
    open("results/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print("\nSaved: results/summary.txt, results/results.json, results/confusion_heldout.png, model_int8.tflite, norm.npz, norm.h")

if __name__ == "__main__":
    main()
