"""selftest.py - Step 3 check. Runs every Python part of the project on made-up signals.
Usage:  python src/selftest.py      (from the project root, virtualenv active)"""
import os, sys, time, io, contextlib, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow chatter
warnings.filterwarnings("ignore")
ok = True
def check(name, cond, info=""):
    global ok
    print(("  OK   " if cond else "  FAIL ") + name + (f"  ({info})" if info else ""))
    ok &= bool(cond)

print("1. Libraries")
import numpy as np
try:
    import serial, serial.tools.list_ports
    check("pyserial", True, serial.__version__)
except ImportError:
    check("pyserial", False, "run: pip install pyserial")
try:
    import tensorflow as tf
    check("tensorflow", True, tf.__version__)
except ImportError:
    tf = None
    check("tensorflow", False, "run: pip install tensorflow")
check("python version", sys.version_info[:2] >= (3, 9), sys.version.split()[0])

print("2. Feature extraction (features.py)")
import features as F, baseline as B
rng = np.random.default_rng(0)
def fake(f_fire, rattle=0.0, n=F.SR):
    t = np.arange(n) / F.SR
    pulses = (np.sin(2 * np.pi * f_fire * t) > 0.98).astype(float)
    k = np.exp(-np.arange(300) / 40.0) * np.sin(2 * np.pi * np.arange(300) * 180 / F.SR)
    x = 2048 + np.convolve(pulses, k, "same") * 60 + rattle * 60 * np.sin(2 * np.pi * 1500 * t) + rng.normal(0, 8, n)
    return np.clip(x, 0, 4095).astype(np.uint16)
m = F.mfcc_window(fake(25))
check("MFCC window shape is 30 x 13", m.shape == (F.N_FRAMES, F.N_MFCC), str(m.shape))
check("MFCC values are finite", np.isfinite(m).all())
est = [round(F.rpm_proxy(fake(f)), 1) for f in (12, 25, 45)]
check("rev proxy tracks 12 / 25 / 45 Hz", all(abs(e - f) < 2 for e, f in zip(est, (12, 25, 45))), str(est))

print("3. Learn-your-bike baseline (baseline.py)")
bl = B.Baseline(1)
for _ in range(40): bl.update(0, F.mfcc_window(fake(25)).mean(0))
h = np.mean([bl.score(0, F.mfcc_window(fake(25)).mean(0)) for _ in range(8)])
a = np.mean([bl.score(0, F.mfcc_window(fake(25, rattle=1.0)).mean(0)) for _ in range(8)])
check("healthy scores low, rattle scores high", h < 3 and a > 3 * h, f"healthy {h:.1f}, rattle {a:.1f}")

print("4. Firmware tables (gen_headers.py)")
import subprocess
r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen_headers.py")], capture_output=True, text=True)
path = os.path.join("firmware", "monitor", "dsp_tables.h")
check("dsp_tables.h written for the ESP32", r.returncode == 0 and os.path.exists(path), r.stdout.strip() or r.stderr.strip()[-120:])

print("5. Model training and int8 conversion (train.py)")
if tf is not None:
    import train as T
    t0 = time.time()
    X = rng.normal(size=(120, F.N_FRAMES, F.N_MFCC, 1)).astype("float32"); y = rng.integers(0, 3, 120)
    model = T.build(); model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=1, verbose=0)
    out = os.path.join("selftest_model.tflite")
    with contextlib.redirect_stdout(io.StringIO()):
        T.to_int8_tflite(model, X, out)
    size = os.path.getsize(out); os.remove(out)
    check("tiny CNN trains and converts to int8", size < 30000, f"{model.count_params()} parameters, {size/1024:.1f} KB, {time.time()-t0:.0f} s")
else:
    check("tiny CNN trains and converts to int8", False, "TensorFlow missing")

print("6. Serial ports (plug the ESP32 in to see it here)")
try:
    import serial.tools.list_ports
    ports = [p.device for p in serial.tools.list_ports.comports()]
    esp = [p for p in ports if "usbserial" in p or "SLAB" in p or "COM" in p]
    print("       found:", ", ".join(ports) if ports else "none")
    print("       ESP32 port:", esp[0] if esp else "not plugged in (fine for now)")
except Exception as e:
    print("       could not list ports:", e)

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED - copy the FAIL lines and send them")
