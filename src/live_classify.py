"""live_classify.py - live CNN prediction on the laptop screen (for the demo).
Needs logger.ino on the ESP32 and a trained model (run pipeline.py first).
  python live_classify.py /dev/cu.usbserial-0001
  python live_classify.py --file data/imbalance/s1_speed1.npy     (replay a recording, no ESP32 needed)"""
import argparse, os, sys, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf
import features as F

def blocks_from_serial(port):
    import serial
    s = serial.Serial(port, 921600, timeout=1); s.reset_input_buffer()
    while True:
        if s.read(1) != b"\xA5" or s.read(1) != b"\x5A": continue
        s.read(1); d = s.read(1024)
        if len(d) == 1024: yield np.frombuffer(d, "<u2") & 0x0FFF

def blocks_from_file(path):
    x = np.load(path)
    for i in range(0, len(x) - 512, 512):
        time.sleep(512 / F.SR); yield x[i:i + 512]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("port", nargs="?"); ap.add_argument("--file"); ap.add_argument("--max-seconds", type=float, default=0)
    a = ap.parse_args()
    if not (a.port or a.file): ap.error("give the serial port or --file")
    nz = np.load("norm.npz"); mu, sd, classes = nz["mu"], nz["sd"], [str(c) for c in nz["classes"]]
    it = tf.lite.Interpreter(model_path="model_int8.tflite"); it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]; s, z = i["quantization"]; os_, oz = o["quantization"]
    buf = np.zeros(F.WIN_SAMPLES, np.uint16); filled = 0; since = 0; t0 = time.time()
    src = blocks_from_file(a.file) if a.file else blocks_from_serial(a.port)
    print("Listening... Ctrl+C to stop\n")
    for b in src:
        buf = np.roll(buf, -len(b)); buf[-len(b):] = b; filled += len(b); since += len(b)
        if filled < F.WIN_SAMPLES or since < F.SR // 2: continue
        since = 0
        x = ((F.mfcc_window(buf) - mu) / sd)[None, ..., None]
        it.set_tensor(i["index"], np.clip(np.round(x / s + z), -128, 127).astype(np.int8)); it.invoke()
        p = (it.get_tensor(o["index"])[0].astype(float) - oz) * os_
        rms = float(np.std(buf.astype(float)))
        line = "  ".join(f"{c[:10]:>10} {'#' * int(round(20 * v)):<20} {100 * v:3.0f}%" for c, v in zip(classes, p))
        sys.stdout.write(f"\r{classes[int(p.argmax())].upper():>12} | {line} | rms {rms:5.1f}   "); sys.stdout.flush()
        if a.max_seconds and time.time() - t0 > a.max_seconds: break
    print()

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\nstopped")
