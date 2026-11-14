"""log_detector.py - record the detector's decisions for the results chapter.
Needs detector.ino on the ESP32 (close the Arduino Serial Monitor first).

  python log_detector.py /dev/cu.usbserial-0001 healthy 120     log 120 s of a healthy run
  python log_detector.py /dev/cu.usbserial-0001 imbalance 60    log 60 s with a fault added
  python log_detector.py --summary                              table of every run so far

Fault runs: switch the fan OFF, add the fault, start this script, then switch the fan ON.
The time from the first running decision to the first ALERT is the detection delay."""
import csv, glob, os, sys, time

def log(port, label, secs):
    import serial
    os.makedirs("results/detector", exist_ok=True)
    path = f"results/detector/{label}_{time.strftime('%H%M%S')}.csv"
    s = serial.Serial(port, 115200, timeout=1); t0 = time.time(); n = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["ms", "state", "score", "threshold", "rms", "votes", "fault", "match"])
        while time.time() - t0 < secs:
            line = s.readline().decode(errors="ignore").strip()
            if line.startswith("D,"):
                parts = line.split(",")[1:]
                if len(parts) >= 6:
                    parts = (parts + ["", ""])[:8]; w.writerow(parts); n += 1
                    print(f"\r{n:4d} decisions  last: {parts[1]:<6} score {parts[2]:>6} {parts[6]:<3}", end="")
            elif line.startswith("#"): print("\n" + line)
    print(f"\nSaved {path}"); summarise([path])

def stats(path):
    rows = list(csv.DictReader(open(path)))
    run = [r for r in rows if r["state"] in ("OK", "WATCH", "ALERT")]
    label = os.path.basename(path).rsplit("_", 1)[0]
    if not run: return label, len(rows), 0, 0, 0, None, None, "-"
    t_on = int(run[0]["ms"]); alerts = [r for r in run if r["state"] == "ALERT"]
    episodes = sum(1 for k, r in enumerate(run) if r["state"] == "ALERT" and (k == 0 or run[k - 1]["state"] != "ALERT"))
    delay = (int(alerts[0]["ms"]) - t_on) / 1000 if alerts else None
    minutes = (int(run[-1]["ms"]) - t_on) / 60000 or 1 / 60
    faults = [r.get("fault") or "" for r in alerts if r.get("fault")]
    fault = max(set(faults), key=faults.count) if faults else "-"
    return label, len(run), 100 * len(alerts) / len(run), episodes, episodes / minutes, delay, minutes, fault

def summarise(paths):
    print("\nrun | decisions | % decisions in ALERT | alert episodes | episodes per minute | first alert after (s) | minutes | suggested fault")
    for p in sorted(paths):
        label, n, pct, ep, epm, d, mins, fault = stats(p)
        print(f"{os.path.basename(p)} | {n} | {pct:.1f}% | {ep} | {epm:.2f} | {'-' if d is None else f'{d:.1f}'} | {0 if mins is None else mins:.1f} | {fault}")

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--summary": summarise(glob.glob("results/detector/*.csv"))
    elif len(sys.argv) == 4: log(sys.argv[1], sys.argv[2], float(sys.argv[3]))
    else: print(__doc__)
