# SonoCrank (EngineSense)

Edge-based motor/engine anomaly detection. A piezo contact microphone picks up engine vibration, an ESP32 turns it into MFCC features on-device, and a small model (CNN classifier or healthy-only autoencoder, TinyML) flags faults such as imbalance, rubbing, chain faults and exhaust leaks. Prototyped on a desk fan before moving to a two-wheeler.

CS2404 (Internet of Things) group project, RV University.

## Project structure

```
SonoCrank/
├── firmware/            # ESP32 Arduino sketches
│   ├── hello/             first upload: blink + chip info
│   ├── bringup_test/      LEDs, buzzer, button, signal level
│   ├── vibe_check/        check the piezo sees the vibration
│   ├── logger/            raw recording to the laptop (used by src/record.py)
│   ├── monitor/           starter for running the CNN on the ESP32 (future work)
│   ├── detector/          on-device learn-your-machine detector + BLE alerts
│   └── detector_v3_backup/  earlier detector version
├── src/                 # Python: recording, training, evaluation, live demos
├── app/                 # Phone/web app (PWA) for the live display
├── hardware/breadboard/ # Breadboard layout and clearance checks
├── data/                # Recordings: data/<condition>/<session>_<name>.npy (not tracked)
├── docs/                # Autoencoder review and figures
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/selftest.py             # checks the Python setup
```

- Arduino IDE: board **ESP32 Dev Module**, esp32 core **2.0.17**.
- **Run every script from the project root** (they read `data/` and write `results/` relative to it), e.g. `python src/pipeline.py`.
- Close the Arduino Serial Monitor before running any script that uses the serial port.
- Recordings, `results/` and trained models are generated locally and are git-ignored.

## Day 1 - record and train
1. Upload firmware/bringup_test -> LEDs, beep, button, signal level OK.
2. Upload firmware/logger. Tape the piezo firmly to the fan's motor housing.
3. Record 60 s per file, 3 sessions x 2 speeds per condition. Remove and redo the fault between sessions.
     python src/record.py <port> 60 data/healthy/s1_speed1.npy
     python src/record.py <port> 60 data/imbalance/s1_speed1.npy     (coin taped to one blade)
     python src/record.py <port> 60 data/rubbing/s1_speed1.npy       (paper strip lightly touching blades)
   ...then s1_speed2, s2_speed1, s2_speed2, s3_speed1, s3_speed2 for each.
   Empty folders (chain_fault, exhaust_leak) are ignored - use them only for bike recordings.
4. python src/pipeline.py      -> results/summary.txt, results/confusion_heldout.png, model_int8.tflite
5. python src/plots.py         -> results/signals.png, results/spectra.png, results/mfcc.png
6. Demo display:  python src/live_classify.py <port>   (logger.ino on the ESP32)

## Healthy-only route (scooter: only healthy recordings available)
1. Upload firmware/logger, close Serial Monitor. Engine idling on the centre stand.
2. One file per session, engine switched off and on between sessions, piezo never moved:
     python src/record.py <port> 60 data/healthy/s1_idle.npy     ... up to s6_idle.npy
   Also once:  python src/record.py <port> 20 test/engine_off.npy   (noise floor)
3. python src/train_healthy.py   -> results/healthy_summary.txt, results/healthy_scores.png, ae_int8.tflite, ae_norm.npz
   Learns healthy only (z-score = ESP32 method, plus a small autoencoder); tests false alerts on held-out
   sessions and detection of SIMULATED faults (rattle, whine, knock at three sizes) added to real healthy data.
4. Demo without the scooter:
     python src/live_anomaly.py --file data/healthy/s3_idle.npy --fault rattle --at 20
   Live from the sensor:  python src/live_anomaly.py <port>
5. Noise floor check:  python src/compare.py test/engine_off.npy data/healthy/s1_idle.npy

## Speed-aware autoencoder on the ESP32
1. python src/export_ae.py          -> firmware/detector/ae_model.h (trained on data/healthy, replayed at 5 speeds)
2. python src/speed_test.py         -> results/speed_summary.txt (one-speed z-score vs speed-aware autoencoder)
3. Upload firmware/detector. It uses the autoencoder by default; type  a  in Serial Monitor to switch to the
   z-score and back (the choice is remembered). Hold the button (learn) once after uploading.
   On a bike: during learning, idle and then hold a few steady rev levels, so idle counts as normal running.

## Day 2 - on-device detector
1. Upload firmware/detector (if "Sketch too big": Tools > Partition Scheme > Huge APP).
2. Serial Monitor 115200: with the fan OFF type  o  (sets the machine-off level). Close the monitor.
3. Fan ON healthy, hold the button 1 s (or type c). Yellow fast blink for ~30 s, then steady.
4. Results runs (close Serial Monitor first):
     python src/log_detector.py <port> healthy_speed1 300
     python src/log_detector.py <port> imbalance_run1 60   (fan OFF, add coin, start script, fan ON)
     python src/log_detector.py --summary
5. Phone: nRF Connect > connect "EngineSense" > Nordic UART Service > TX > enable notifications (show as text).

## Python scripts (`src/`)

| Script | What it does |
|---|---|
| `record.py` | Record raw ADC counts from the logger sketch to `.npy` |
| `features.py`, `baseline.py` | MFCC reference (must match the firmware) and baseline maths |
| `train.py`, `pipeline.py` | CNN training, held-out-session testing, int8 export |
| `plots.py` | Report figures from recordings |
| `live_classify.py` | Live CNN display for the demo |
| `anomaly.py`, `train_healthy.py` | Healthy-only training/testing with simulated faults |
| `live_anomaly.py` | Live or replayed anomaly display (healthy-only model) |
| `export_ae.py` | Train the speed-aware autoencoder, write `firmware/detector/ae_model.h` |
| `speed_test.py` | Variable-speed test of the detectors |
| `gen_headers.py`, `gen_faults.py` | Generate firmware headers (`dsp_tables.h`, `faults.h`) |
| `log_detector.py` | Log detector decisions; summary table for the report |
| `compare.py` | Compare recordings by frequency content |
| `ae_review_prep.py`, `ae_review_exp.py` | Autoencoder review experiments (see `docs/autoencoder_review.md`) |
| `selftest.py` | Check the Python setup |

Breadboard check: `python hardware/breadboard/clearance.py`
