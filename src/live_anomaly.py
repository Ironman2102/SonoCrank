"""live_anomaly.py - live 'is this still healthy?' display on the laptop (demo).
  python live_anomaly.py /dev/cu.usbserial-0001                          live from the sensor (logger.ino on the ESP32)
  python live_anomaly.py --file data/healthy/s1_idle.npy --fault rattle --at 20
        replays a real recording; from second 20 a simulated rattle is added (no scooter needed)
Run train_healthy.py first."""
import argparse, os, sys, time, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2"); warnings.filterwarnings("ignore")
import numpy as np, tensorflow as tf, features as F, anomaly as A
from live_classify import blocks_from_serial

ap = argparse.ArgumentParser(); ap.add_argument("port", nargs="?"); ap.add_argument("--file")
ap.add_argument("--fault", choices=A.FAULTS); ap.add_argument("--at", type=float, default=20); ap.add_argument("--severity", type=float, default=1.0)
a = ap.parse_args()
if not (a.port or a.file): ap.error("give the serial port or --file")
nz = np.load("ae_norm.npz"); it = tf.lite.Interpreter(model_path="ae_int8.tflite"); it.allocate_tensors()
i, o = it.get_input_details()[0], it.get_output_details()[0]; s, zp = i["quantization"]; os_, oz = o["quantization"]
def ae_score(W):
    x = (A.ae_features(W[None])[0] - nz["mu"]) / nz["sd"]
    it.set_tensor(i["index"], np.clip(np.round(x / s + zp), -128, 127).astype(np.int8)[None]); it.invoke()
    y = (it.get_tensor(o["index"])[0].astype(float) - oz) * os_; return float(np.sqrt(((y - x) ** 2).mean()))
def blocks_file(path):
    x = np.load(path)
    if a.fault:
        k = int(a.at * F.SR); x = np.concatenate([x[:k], A.inject(x[k:], a.fault, a.severity, seed=3)])
    for j in range(0, len(x) - 512, 512):
        time.sleep(512 / F.SR); yield x[j:j + 512]
src = blocks_file(a.file) if a.file else blocks_from_serial(a.port)
buf = np.zeros(F.WIN_SAMPLES, np.uint16); filled = since = 0; hist = []; t = 0.0; thr = float(nz["thr"])
print(f"threshold {thr:.2f}  (Ctrl+C to stop)\n")
try:
    for b in src:
        buf = np.roll(buf, -len(b)); buf[-len(b):] = b; filled += len(b); since += len(b); t += len(b) / F.SR
        if filled < F.WIN_SAMPLES or since < F.SR // 2: continue
        since = 0; sc = ae_score(F.mfcc_window(buf)); hist = (hist + [sc > thr])[-6:]
        state = "ALERT" if sum(hist) >= 4 else ("WATCH" if hist[-1] else "OK")
        bar = "#" * min(40, int(20 * sc / thr))
        sys.stdout.write(f"\r{t:6.1f} s  {state:5s}  score {sc:6.2f}  |{bar:<40}|   "); sys.stdout.flush()
        if state == "ALERT" and (len(hist) < 2 or sum(hist[:-1]) < 4): sys.stdout.write("\n")
except KeyboardInterrupt: pass
print()
