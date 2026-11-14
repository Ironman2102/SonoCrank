// monitor_skeleton.ino - Phase 5/6 STARTER. Structure + the tricky parts; fill TODOs and verify on the bench.
// Files next to the sketch: dsp.h, dsp_tables.h (gen_headers.py), norm.h + model_data.h (pipeline.py + xxd)
#include <Arduino.h>
#include <Preferences.h>
#include "driver/i2s.h"
#include "driver/adc.h"
#include "dsp.h"
#include "norm.h"
#include "model_data.h"                 // const unsigned char model_int8_tflite[]  (name from xxd)
// TFLM headers - exact include paths depend on the Arduino library you installed:
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define SR 16000
#define N_FRAMES 30
#define PIN_RED 26
#define PIN_GRN 27
#define PIN_BUZ 25
#define PIN_BTN 14                      // hold 3 s = calibration mode

static uint16_t ring[SR];               // last 1 s of raw counts (32 KB)
static uint32_t wpos = 0;               // total samples written
static float frames[N_FRAMES][N_MFCC];  // rolling MFCC frames
static int   nFrames = 0, newFrames = 0;

// ---- TFLM -----------------------------------------------------------------
constexpr int kArena = 40 * 1024;       // raise if AllocateTensors() fails; print arena_used_bytes() and trim
static uint8_t arena[kArena];
static tflite::MicroMutableOpResolver<6> resolver;
static tflite::MicroInterpreter* interp;
static TfLiteTensor *tin, *tout;

void tflmBegin() {
  resolver.AddConv2D(); resolver.AddMaxPool2D(); resolver.AddMean();
  resolver.AddFullyConnected(); resolver.AddSoftmax(); resolver.AddReshape();   // add more if "Didn't find op" appears
  static tflite::MicroInterpreter si(tflite::GetModel(model_int8_tflite), resolver, arena, kArena);  // older libs also want an error reporter
  interp = &si;
  if (interp->AllocateTensors() != kTfLiteOk) { Serial.println("AllocateTensors failed"); while (1); }
  tin = interp->input(0); tout = interp->output(0);
  Serial.printf("arena used: %u bytes\n", (unsigned)interp->arena_used_bytes());
}

// frames -> int8 input -> Invoke -> probabilities[3]; returns inference time in microseconds
uint32_t classify(float* prob) {
  int8_t* in = tin->data.int8;
  int start = nFrames - N_FRAMES, k = 0;
  for (int f = 0; f < N_FRAMES; f++) for (int c = 0; c < N_MFCC; c++) {
    float v = (frames[(start + f) % N_FRAMES][c] - MFCC_MEAN[c]) / MFCC_STD[c];
    int q = (int)roundf(v / tin->params.scale) + tin->params.zero_point;
    in[k++] = (int8_t)constrain(q, -128, 127);
  }
  uint32_t t0 = micros(); interp->Invoke(); uint32_t dt = micros() - t0;
  for (int i = 0; i < 3; i++) prob[i] = (tout->data.int8[i] - tout->params.zero_point) * tout->params.scale;
  return dt;
}

// ---- Novelty: RPM proxy (mirror of features.rpm_proxy) + baseline ----------
float rpmProxyHz() {                    // dominant periodicity of the 1 kHz vibration envelope
  static float env[1000]; float mean = 0;
  for (int i = 0; i < 1000; i++) { float s = 0; for (int j = 0; j < 16; j++) s += fabsf((float)ring[(wpos + i * 16 + j) % SR] - 2048.0f); env[i] = s / 16; mean += env[i]; }
  mean /= 1000; for (int i = 0; i < 1000; i++) env[i] -= mean;
  float best = -1e30; int bl = 10;
  for (int l = 10; l <= 100; l++) { float a = 0; for (int i = 0; i + l < 1000; i++) a += env[i] * env[i + l]; if (a > best) { best = a; bl = l; } }
  return 1000.0f / bl;
}
int rpmBin(float hz) { return hz < 20 ? 0 : (hz < 35 ? 1 : 2); }   // TODO: replace edges with terciles from YOUR healthy data

struct Baseline { float n[3] = {0}, mu[3][N_MFCC] = {{0}}, m2[3][N_MFCC] = {{0}};
  void update(int b, const float* v) { n[b]++; for (int c = 0; c < N_MFCC; c++) { float d = v[c] - mu[b][c]; mu[b][c] += d / n[b]; m2[b][c] += d * (v[c] - mu[b][c]); } }
  float score(int b, const float* v) { float s = 0; for (int c = 0; c < N_MFCC; c++) { float sd = sqrtf(m2[b][c] / max(n[b] - 1, 1.0f)) + 1e-3f; float z = (v[c] - mu[b][c]) / sd; s += z * z; } return sqrtf(s / N_MFCC); }
} baseline;
// TODO: save/load baseline with Preferences (putBytes / getBytes) so calibration survives power-off.

// ---- main -----------------------------------------------------------------
void setup() {
  Serial.begin(115200); dsp_init(); tflmBegin();
  pinMode(PIN_RED, OUTPUT); pinMode(PIN_GRN, OUTPUT); pinMode(PIN_BUZ, OUTPUT); pinMode(PIN_BTN, INPUT_PULLUP);
  /* TODO: i2s/adc init exactly as in logger.ino */
}

void loop() {
  uint16_t blk[HOP]; size_t got = 0;
  i2s_read(I2S_NUM_0, blk, sizeof(blk), &got, portMAX_DELAY);
  for (int i = 0; i < HOP; i++) ring[(wpos + i) % SR] = blk[i] & 0x0FFF;
  wpos += HOP;
  if (wpos < N_FFT) return;

  uint16_t fr[N_FFT];                                           // newest 1024 samples = one frame per 512-sample hop
  for (int i = 0; i < N_FFT; i++) fr[i] = ring[(wpos - N_FFT + i) % SR];
  mfcc_frame(fr, frames[nFrames % N_FRAMES]); nFrames++; newFrames++;

  if (nFrames >= N_FRAMES && newFrames >= 15) {                 // decision every ~0.5 s
    newFrames = 0;
    float rms = 0;  /* TODO: engine-off gate: compute RMS of counts; if below threshold -> idle state, return */
    float p[3]; uint32_t us = classify(p);
    float meanMfcc[N_MFCC] = {0};
    for (int f = 0; f < N_FRAMES; f++) for (int c = 0; c < N_MFCC; c++) meanMfcc[c] += frames[f][c] / N_FRAMES;
    int bin = rpmBin(rpmProxyHz()); float score = baseline.score(bin, meanMfcc);
    // TODO: vote over last 6 decisions (>=4 faulty -> alert); drive LEDs/buzzer; BLE notify; SD log; Serial.printf for debugging
    Serial.printf("p=[%.2f %.2f %.2f] score=%.2f bin=%d infer=%luus\n", p[0], p[1], p[2], score, bin, (unsigned long)us);
  }
}
