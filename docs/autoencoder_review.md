# Autoencoder review: findings and fix

Review of the healthy-only autoencoder (AE) in `claude/EngineSense_Project_Reference.md` §6.2, following `claude/Autoencoder_Review_Plan.md`.
**Data:** the 18 real fan recordings. **Method:** leave-one-session-out, healthy-only training, threshold = 1.5 × max training score, alert = 4 of the last 6 windows (same rules as the existing code).
**Code:** `src/anomaly.py`, `src/train_healthy.py`, `src/features.py` (Mac copies identical to the reviewed ones). Experiment script: `src/ae_review_exp.py`.

## 1. Verdict on the suspected causes
| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 1 | Per-window normalisation erased loudness | **Rejected** | The AE normalised each of the 390 inputs with mean/std from the healthy *training* set, not per window. The imbalance shift reached the AE input: c0 at +2.8 sd |
| 2 | c0 diluted in a 390-value mean | **Not the cause** | c0 is 1/13 of the score in both the AE and the z-score. Weighting the worst coefficient only raised imbalance to 94.5% |
| 3 | AE reconstructs the anomaly | **Rejected** | The AE did *not* reconstruct the shift: c0 input +2.82 → residual −2.63, c3/c4 ≈ −1.0 |
| 4 | Unequal comparison: per-frame vs window mean | **Confirmed: the main cause** | Per-frame MFCC noise is **3–6× larger** than the noise of the 30-frame mean. The same imbalance shift is 14 sd for the z-score but only 2.8 sd per frame for the AE. The frame noise sets the AE's healthy error at ~0.9–1.1 and its threshold at 1.67, so the imbalance error (median 1.35) stays below it in every session |
| 5 | Knock transient averaged away | **Rejected (different cause)** | Max-over-frames scoring did *not* help (0% at all sizes). The simulated knock raises only the two lowest mel bands (62 and 125 Hz) by **< 1 frame-sd** at medium size, so it is masked by the fan's own low-frequency vibration (rotation ≈ 19–22 Hz) |

## 2. Experiment results (mean of 6 held-out sessions, % of time alerting)
| Detector | False alerts | Real imbalance | Real rubbing | Rattle s/m/l | Whine s/m/l | Knock s/m/l |
|---|---|---|---|---|---|---|
| RMS only (loudness baseline) | 0.0 | 97.5 | 75.5 | 0/0/0 | 0/0/0 | 0/0/74 |
| z-score (ESP32) | 0.0 | 97.5 | 97.5 | 5/79/97 | 97/97/97 | 0/0/34 |
| z-score, worst frame | 0.0 | 0.0 | 37.3 | 0/4/97 | 97/97/97 | 0/0/0 |
| **AE old (390 inputs, mean error)** | 0.0 | **0.0** | 92.7 | 5/85/97 | 97/97/97 | 0/0/0 |
| AE frame-averaged residual (#1/#4 fix) | 0.0 | 97.5 | 97.5 | 10/74/95 | 97/97/97 | 0/0/61 |
| AE latent Mahalanobis (#2) | 0.0 | 0.0 | 0.0 | 0/0/0 | 0/0/14 | 0/0/0 |
| AE max-over-frames (#3) | 0.0 | 0.0 | 56.4 | 9/65/96 | 97/97/97 | 0/0/0 |
| AE worst coefficient (#4) | 0.0 | 94.5 | 91.6 | 3/81/97 | 97/97/97 | 0/0/0 |
| **AE-26: mean+std inputs, 26-16-6-16-26 (#5)** | **0.0** | **97.5** | **97.5** | **54/97/97** | 97/97/97 | 0/0/38 |
| Hybrid z OR old AE (#6) | 0.0 | 97.5 | 97.5 | 10/88/97 | 97/97/97 | 0/0/34 |
| Hybrid z OR frame-averaged AE (#6) | 0.0 | 97.5 | 97.5 | 11/83/97 | 97/97/97 | 0/0/70 |

97.5% is the ceiling (the first windows of a fault still contain healthy sound). Time to first alert on real faults: 2.5 s for every detector that catches them. AE-26 gave identical results with seeds 0, 1 and 2.

## 3. Decision (applied)
- **The laptop AE is now AE-26** (`anomaly.py`): inputs are the window mean and std of the 13 MFCCs, normalised from healthy training data. It has **0% false alerts, 97.5% real imbalance and rubbing**, and is the best detector on small and medium rattle (54% / 97% vs z-score 5% / 79%). It has about 1,100 parameters (5.5 KB int8), so it also fits on the ESP32.
- `train_healthy.py` and `live_anomaly.py` were updated, `results/healthy_*` regenerated, and the live replay checked.
- The ESP32 keeps the z-score (equal on real faults, no model needed). Running AE-26 on the ESP32 is a realistic next step.

## 4. Honest caveats for the report and viva
- **The real faults on this rig are mostly louder.** A plain RMS threshold already catches 97.5% of real imbalance (and 75.5% of rubbing). The MFCC detectors add rubbing to 97.5% and catch spectral changes that leave loudness unchanged (whine: RMS 0% vs 97%). Real imbalance results therefore partly measure loudness.
- **Knock is not detected at small or medium size by any MFCC detector.** The cause is masking in the 20–125 Hz region and coarse low-frequency resolution: 15.6 Hz FFT bins, first mel filter centred near 62 Hz, fan rotation ≈ 20 Hz below it. Fix (future work): a low-frequency band or envelope/order analysis, or a lower F_MIN with a longer FFT.
- Each fault was attached once per condition, so the "unseen session" results partly measure one putty or paper placement.
- **Mount sensitivity (not yet tested):** retesting a healthy fan with a looser or tighter mount would show whether a mounting change alone triggers alerts. Re-learning after re-mounting (the button) is the intended procedure.
- The AE's real advantage should come with **variable engine speed (bike)**, where a fixed mean/std baseline is too broad. A single-speed rig cannot show that, and on this rig the AE mainly matches the z-score.
