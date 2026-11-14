"""features.py - MUST stay identical (same constants, same maths) to the C++ firmware."""
import numpy as np

SR = 16000            # sample rate (Hz)
N_FFT = 1024          # 64 ms frame, 15.6 Hz per bin (needed to resolve engine low end)
HOP = 512             # 32 ms hop (50 % overlap)
N_MEL = 24
N_MFCC = 13
F_MIN, F_MAX = 20.0, 4000.0
WIN_SAMPLES = SR      # 1 s analysis window
N_FRAMES = 1 + (WIN_SAMPLES - N_FFT) // HOP   # = 30
ADC_MID, ADC_SCALE = 2048.0, 2048.0           # 12-bit ADC counts -> roughly -1..+1

def _hz2mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
def _mel2hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def mel_filterbank():
    hz = _mel2hz(np.linspace(_hz2mel(F_MIN), _hz2mel(F_MAX), N_MEL + 2))
    b = np.floor((N_FFT + 1) * hz / SR).astype(int)
    fb = np.zeros((N_MEL, N_FFT // 2 + 1), np.float32)
    for m in range(1, N_MEL + 1):
        l, c, r = b[m - 1], b[m], b[m + 1]
        for k in range(l, c): fb[m - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r): fb[m - 1, k] = (r - k) / max(r - c, 1)
    return fb

def dct_matrix():
    n = np.arange(N_MEL)
    d = np.array([np.sqrt(2.0 / N_MEL) * np.cos(np.pi * k * (2 * n + 1) / (2 * N_MEL))
                  for k in range(N_MFCC)], np.float32)
    d[0] /= np.sqrt(2.0)
    return d

_FB, _DCT, _HANN = mel_filterbank(), dct_matrix(), np.hanning(N_FFT).astype(np.float32)

def mfcc_window(counts):
    """counts: 1-D array of raw 12-bit ADC counts, length >= WIN_SAMPLES -> (N_FRAMES, N_MFCC)"""
    x = (np.asarray(counts[:WIN_SAMPLES], np.float32) - ADC_MID) / ADC_SCALE
    out = np.zeros((N_FRAMES, N_MFCC), np.float32)
    for i in range(N_FRAMES):
        f = x[i * HOP: i * HOP + N_FFT]
        f = (f - f.mean()) * _HANN
        p = (np.abs(np.fft.rfft(f)) ** 2) / N_FFT
        logmel = np.log(_FB @ p + 1e-10)
        out[i] = _DCT @ logmel
    return out

def rpm_proxy(counts, lo_ms=10, hi_ms=100):
    """Dominant periodicity of the vibration envelope -> frequency in Hz (firing/rotation proxy)."""
    x = np.abs((np.asarray(counts[:WIN_SAMPLES], np.float32) - ADC_MID))
    env = x[: (len(x) // 16) * 16].reshape(-1, 16).mean(axis=1)      # 1 kHz envelope
    env = env - env.mean()
    ac = np.array([np.dot(env[:-l], env[l:]) for l in range(lo_ms, hi_ms + 1)])
    lag = lo_ms + int(np.argmax(ac))
    return 1000.0 / lag
