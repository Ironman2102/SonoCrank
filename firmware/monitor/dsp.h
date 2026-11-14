// dsp.h - on-device MFCC frame (mirrors features.py). Pure C++, no libraries.
#pragma once
#include <math.h>
#include <stdint.h>
#include "dsp_tables.h"
#define N_FFT 1024
#define HOP   512
#define N_MEL 24
#define N_MFCC 13

static float    tw_re[N_FFT / 2], tw_im[N_FFT / 2];
static uint16_t bitrev[N_FFT];

inline void dsp_init() {
  for (int i = 0; i < N_FFT / 2; i++) { tw_re[i] = cosf(-2.0f * (float)M_PI * i / N_FFT); tw_im[i] = sinf(-2.0f * (float)M_PI * i / N_FFT); }
  for (int i = 0; i < N_FFT; i++) { int r = 0; for (int b = 0; b < 10; b++) if (i & (1 << b)) r |= 1 << (9 - b); bitrev[i] = r; }
}

static void fft1024(float* re, float* im) {
  for (int i = 0; i < N_FFT; i++) { int j = bitrev[i]; if (j > i) { float t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
  for (int len = 2; len <= N_FFT; len <<= 1) {
    int half = len >> 1, step = N_FFT / len;
    for (int i = 0; i < N_FFT; i += len)
      for (int k = 0; k < half; k++) {
        float wr = tw_re[k * step], wi = tw_im[k * step]; int a = i + k, b = a + half;
        float tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
      }
  }
}

// counts: N_FFT raw 12-bit ADC samples  ->  out: N_MFCC coefficients
inline void mfcc_frame(const uint16_t* counts, float* out) {
  static float re[N_FFT], im[N_FFT];
  float mean = 0;
  for (int i = 0; i < N_FFT; i++) { re[i] = ((float)counts[i] - 2048.0f) / 2048.0f; mean += re[i]; }
  mean /= N_FFT;
  for (int i = 0; i < N_FFT; i++) { re[i] = (re[i] - mean) * HANN[i]; im[i] = 0; }
  fft1024(re, im);
  float lm[N_MEL];
  for (int m = 0; m < N_MEL; m++) {
    float s = 0;
    for (int j = 0; j < MEL_LEN[m]; j++) { int k = MEL_START[m] + j; s += MEL_W[MEL_OFF[m] + j] * (re[k] * re[k] + im[k] * im[k]) / N_FFT; }
    lm[m] = logf(s + 1e-10f);
  }
  for (int c = 0; c < N_MFCC; c++) { float a = 0; for (int m = 0; m < N_MEL; m++) a += DCT[c * N_MEL + m] * lm[m]; out[c] = a; }
}
