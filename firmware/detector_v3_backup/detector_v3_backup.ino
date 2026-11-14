// BACKUP: detector v3 (z-score only), known to work on the fan. Use only if the new detector misbehaves.
// detector.ino - on-device "learn-your-machine" anomaly detector.
//   1. Hold the button for 1 s (or type c in Serial Monitor) while the machine runs normally.
//   2. The ESP32 records ~30 s of healthy vibration and learns its sound (saved across resets).
//   3. From then on it scores every 0.5 s of sound against what it learned and raises an alert
//      (red LED + buzzer + Bluetooth message) when the sound stays abnormal.
// Board: "ESP32 Dev Module", esp32 core 2.0.17.
// If upload says "Sketch too big": Tools -> Partition Scheme -> "Huge APP (3MB No OTA/1MB SPIFFS)".
// Phone: install "nRF Connect", connect to "EngineSense", open Nordic UART Service,
//        enable notifications on the TX characteristic and show values as text (UTF-8).
#include <Arduino.h>
#include <Preferences.h>
#include "driver/i2s.h"
#include "driver/adc.h"
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLE2902.h>
#include "dsp.h"                       // on-device MFCC, identical to features.py
#include "faults.h"                    // fault fingerprints (made by gen_faults.py) for the fault suggestion

// ---------------- pins and settings ----------------
#define PIN_BTN 14
#define PIN_YEL 27
#define PIN_RED 26
#define PIN_BUZ 25
#define SR          16000
#define WIN_FRAMES  30        // 30 frames x 32 ms = about 1 s of sound per decision
#define STEP_FRAMES 15        // one decision every 15 frames = about 0.5 s
#define CAL_WINDOWS 60        // 60 decisions = about 30 s of calibration
#define VOTES       6         // look at the last 6 decisions (3 s)
#define ALERT_VOTES 4         // alert when 4 of the last 6 are abnormal
#define CLEAR_VOTES 1         // clear when at most 1 of the last 6 is abnormal
#define SD_FLOOR    0.10f     // smallest spread allowed per MFCC coefficient
#define THR_MIN     3.0f      // smallest alert threshold allowed
#define RUN_FRAC    0.60f     // judge only when vibration is at least 60 % of the learned running level
#define SETTLE      4         // after reaching running level, wait 4 more decisions (2 s) before judging
#define FALL_FRAC   0.75f     // below 75 % of the running level AND still falling = slowing down
#define MATCH_MIN   0.50f     // a fault must match its fingerprint at least this well, otherwise "unknown"
#define CAL_TIMEOUT 120000    // give up learning after 2 minutes if the machine never runs

const char* NUS_SVC = "6e400001-b5a3-f393-e0a9-e50e24dcca9e";   // Nordic UART service
const char* NUS_TX  = "6e400003-b5a3-f393-e0a9-e50e24dcca9e";   // notify characteristic

// ---------------- state ----------------
enum State { NOCAL, CALIB, OFF, OK_, WATCH, ALERT, SPIN };
const char* STATE_NAME[] = {"NOCAL", "CALIB", "OFF", "OK", "WATCH", "ALERT", "SPIN"};   // SPIN = starting up or slowing down
State state = NOCAL;

struct Baseline { uint32_t magic; float mu[N_MFCC]; float sd[N_MFCC]; float thr; float offRms; float runRms; } base;
const uint32_t MAGIC = 0xE5C0DE03;      // changed when the saved format changes: forces a new calibration
float offRms = 12.0f;                 // ADC counts; quieter than this = machine off

static uint16_t prevHalf[HOP], frame[N_FFT], blk[HOP];
static float mf[WIN_FRAMES][N_MFCC];
static int fIdx = 0, nFrames = 0, sinceDecision = 0;
static double sumV = 0, sumSq = 0; static long nS = 0;    // for the RMS of the current window
static float calVec[CAL_WINDOWS][N_MFCC]; static float calRms[CAL_WINDOWS]; static int calN = 0;
static int settle = 0; static float prevRms = 0;
static uint32_t calStart = 0, calNote = 0; static State calPrev = NOCAL;
static int diagCount[N_FAULTS + 1]; static int diag = -2; static float diagMatch = 0;   // -2 none, -1 unknown, else fault index
static uint8_t votes[VOTES]; static int vIdx = 0;
static bool muted = false;
static uint32_t btnDown = 0; static bool btnFired = false;
static uint32_t lastBle = 0;
static float lastScore = 0, lastRms = 0;

Preferences prefs;
BLECharacteristic* tx = nullptr;
bool bleConnected = false;

class ServerCB : public BLEServerCallbacks {
  void onConnect(BLEServer* s) { bleConnected = true; }
  void onDisconnect(BLEServer* s) { bleConnected = false; s->startAdvertising(); }
};

void bleSend(const char* msg) {
  if (!tx || !bleConnected) return;
  tx->setValue((uint8_t*)msg, strlen(msg));
  tx->notify();
}

// ---------------- helpers ----------------
void setupAdc() {
  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_ADC_BUILT_IN);
  cfg.sample_rate = SR;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = 8;
  cfg.dma_buf_len = HOP;
  cfg.use_apll = false;
  i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL);
  adc1_config_width(ADC_WIDTH_BIT_12);
  adc1_config_channel_atten(ADC1_CHANNEL_6, ADC_ATTEN_DB_11);   // GPIO34
  i2s_set_adc_mode(ADC_UNIT_1, ADC1_CHANNEL_6);
  i2s_adc_enable(I2S_NUM_0);
}

void setupBle() {
  BLEDevice::init("EngineSense");
  BLEServer* srv = BLEDevice::createServer();
  srv->setCallbacks(new ServerCB());
  BLEService* svc = srv->createService(NUS_SVC);
  tx = svc->createCharacteristic(NUS_TX, BLECharacteristic::PROPERTY_NOTIFY | BLECharacteristic::PROPERTY_READ);
  tx->addDescriptor(new BLE2902());
  svc->start();
  BLEAdvertising* adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(NUS_SVC);
  adv->setScanResponse(true);
  BLEDevice::startAdvertising();
}

float score(const float* v) {           // RMS z-score against the learned healthy sound
  float s = 0;
  for (int c = 0; c < N_MFCC; c++) { float z = (v[c] - base.mu[c]) / base.sd[c]; s += z * z; }
  return sqrtf(s / N_MFCC);
}

int diagnose(const float* v, float* match) {   // which known fault does this change of sound look like?
  float z[N_MFCC], nz = 0;
  for (int c = 0; c < N_MFCC; c++) { z[c] = (v[c] - base.mu[c]) / FAULT_SD[c]; nz += z[c] * z[c]; }
  nz = sqrtf(nz) + 1e-6f; int best = -1; float bestCos = -2;
  for (int f = 0; f < N_FAULTS; f++) {
    float d = 0; for (int c = 0; c < N_MFCC; c++) d += z[c] * FAULT_DIR[f][c];
    if (d / nz > bestCos) { bestCos = d / nz; best = f; }
  }
  *match = bestCos;
  return bestCos >= MATCH_MIN ? best : -1;
}
const char* diagCode() { return diag == -2 ? "" : diag == -1 ? "UNK" : FAULT_CODE[diag]; }
void resetDiag() { memset(diagCount, 0, sizeof(diagCount)); diag = -2; diagMatch = 0; }
void sendStatus() {
  char m[24];
  if (diag != -2 && (state == WATCH || state == ALERT)) snprintf(m, sizeof(m), "%s %.1f/%.1f %s", STATE_NAME[state], lastScore, base.thr, diagCode());
  else snprintf(m, sizeof(m), "%s %.1f/%.1f", STATE_NAME[state], lastScore, base.thr);
  bleSend(m);
}

void startCalibration() {
  if (state != CALIB) calPrev = state;
  calN = 0; state = CALIB; calStart = calNote = millis(); resetDiag();
  memset(votes, 0, sizeof(votes));
  Serial.println("# calibrating: keep the machine running normally for about 30 s");
  bleSend("CALIBRATING");
}

void finishCalibration() {
  for (int c = 0; c < N_MFCC; c++) {
    double m = 0, v = 0;
    for (int i = 0; i < calN; i++) m += calVec[i][c];
    m /= calN;
    for (int i = 0; i < calN; i++) v += (calVec[i][c] - m) * (calVec[i][c] - m);
    base.mu[c] = m;
    base.sd[c] = fmaxf(sqrtf(v / (calN - 1)), SD_FLOOR);
  }
  float mx = 0;
  for (int i = 0; i < calN; i++) mx = fmaxf(mx, score(calVec[i]));
  base.thr = fmaxf(THR_MIN, 1.5f * mx);   // 50 % above the worst healthy window
  base.offRms = offRms;
  float r[CAL_WINDOWS]; memcpy(r, calRms, sizeof(float) * calN);     // typical running level = median RMS
  for (int i = 1; i < calN; i++) for (int j = i; j > 0 && r[j] < r[j - 1]; j--) { float t = r[j]; r[j] = r[j - 1]; r[j - 1] = t; }
  base.runRms = r[calN / 2];
  base.magic = MAGIC;
  prefs.putBytes("base", &base, sizeof(base));
  state = OK_; settle = SETTLE;          // the machine is already running steadily
  memset(votes, 0, sizeof(votes));
  Serial.printf("# calibration done: %d windows, worst healthy score %.2f, alert threshold %.2f, running level %.1f\n", calN, mx, base.thr, base.runRms);
  bleSend("CALIBRATED");
}

void handleSerial() {
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == 'c') startCalibration();
    else if (ch == 'x') { prefs.remove("base"); state = NOCAL; Serial.println("# calibration cleared"); }
    else if (ch == 'm') { muted = !muted; Serial.printf("# buzzer %s\n", muted ? "muted" : "on"); }
    else if (ch == 'o') { offRms = lastRms * 1.5f + 1; base.offRms = offRms; if (base.magic == MAGIC) prefs.putBytes("base", &base, sizeof(base));
                          Serial.printf("# machine-off level set to %.1f (do this with the machine OFF)\n", offRms); }
    else if (ch == '?') Serial.println("# c = calibrate, x = clear calibration, o = set machine-off level, m = mute buzzer");
  }
}

void handleButton(uint32_t now) {
  bool down = digitalRead(PIN_BTN) == LOW;
  if (down) {
    if (!btnDown) btnDown = now;
    if (!btnFired && now - btnDown >= 1000) { btnFired = true; startCalibration(); }
  } else { btnDown = 0; btnFired = false; }
}

void driveOutputs(uint32_t now) {
  bool yel = false, red = false, buz = false;
  switch (state) {
    case NOCAL: yel = (now / 500) % 2; break;               // slow blink: needs calibration
    case CALIB: yel = (now / 100) % 2; break;               // fast blink: learning
    case OFF:   yel = (now % 2000) < 200; break;            // clear blink every 2 s: device on, machine off
    case SPIN:  yel = (now % 1000) < 200; break;            // blink every 1 s: starting up or slowing down
    case OK_:   yel = true; break;                          // steady: healthy
    case WATCH: yel = (now / 250) % 2; break;               // blink: something changing
    case ALERT: red = true; buz = !muted && (now % 1000) < 150; break;
  }
  digitalWrite(PIN_YEL, yel); digitalWrite(PIN_RED, red); digitalWrite(PIN_BUZ, buz);
}

// ---------------- main ----------------
void setup() {
  Serial.begin(115200);
  pinMode(PIN_BTN, INPUT_PULLUP);
  pinMode(PIN_YEL, OUTPUT); pinMode(PIN_RED, OUTPUT); pinMode(PIN_BUZ, OUTPUT);
  dsp_init();
  prefs.begin("detector", false);
  if (prefs.getBytesLength("base") == sizeof(base)) {
    prefs.getBytes("base", &base, sizeof(base));
    if (base.magic == MAGIC) { state = OK_; offRms = base.offRms; }
  }
  setupAdc();
  setupBle();
  Serial.printf("# EngineSense ready. %s. Type ? for commands.\n",
                state == NOCAL ? "Not calibrated: hold the button 1 s" : "Calibration loaded");
  Serial.println("# columns: D,ms,state,score,threshold,rms,abnormal_votes,fault,match");
}

void loop() {
  size_t got = 0;
  i2s_read(I2S_NUM_0, blk, sizeof(blk), &got, portMAX_DELAY);
  if (got != sizeof(blk)) return;
  uint32_t now = millis();

  // one MFCC frame per 512 new samples (previous half + new half = 1024)
  for (int i = 0; i < HOP; i++) {
    uint16_t v = blk[i] & 0x0FFF;
    frame[i] = prevHalf[i]; frame[HOP + i] = v; prevHalf[i] = v;
    sumV += v; sumSq += (double)v * v; nS++;
  }
  mfcc_frame(frame, mf[fIdx]);
  fIdx = (fIdx + 1) % WIN_FRAMES;
  if (nFrames < WIN_FRAMES) nFrames++;

  handleSerial();
  handleButton(now);
  driveOutputs(now);

  if (nFrames < WIN_FRAMES || ++sinceDecision < STEP_FRAMES) return;
  sinceDecision = 0;

  // window summary: mean MFCC over the last ~1 s and RMS of the last ~0.5 s
  float v[N_MFCC] = {0};
  for (int f = 0; f < WIN_FRAMES; f++) for (int c = 0; c < N_MFCC; c++) v[c] += mf[f][c] / WIN_FRAMES;
  double mean = sumV / nS; lastRms = sqrt(fmax(0.0, sumSq / nS - mean * mean));
  sumV = sumSq = 0; nS = 0;

  if (state == CALIB) {
    if (lastRms >= offRms) { calRms[calN] = lastRms; memcpy(calVec[calN++], v, sizeof(v)); }   // learn only while running
    if (calN >= CAL_WINDOWS) finishCalibration();
    else if (calN == 0 && now - calNote > 5000) { calNote = now; Serial.println("# calibration waiting: switch the machine on (it learns only while running)"); }
    if (state == CALIB && now - calStart > CAL_TIMEOUT && calN < CAL_WINDOWS) {
      state = (base.magic == MAGIC) ? calPrev : NOCAL; if (state == CALIB) state = NOCAL;
      Serial.println("# calibration cancelled: the machine did not run long enough"); bleSend("CALIB CANCELLED"); return; }
    Serial.printf("D,%lu,CALIB,0,0,%.1f,%d\n", (unsigned long)now, lastRms, calN);
    return;
  }
  if (state == NOCAL) { Serial.printf("D,%lu,NOCAL,0,0,%.1f,0\n", (unsigned long)now, lastRms); return; }

  int abnormal = 0;
  State before = state;
  float runGate = fmaxf(offRms, RUN_FRAC * base.runRms);
  bool slowing = lastRms < FALL_FRAC * base.runRms && lastRms < 0.9f * prevRms;   // weaker than normal and dropping
  prevRms = lastRms;
  if (lastRms < runGate || slowing) {        // off, starting up or slowing down: do not judge
    state = lastRms < offRms ? OFF : SPIN;
    memset(votes, 0, sizeof(votes)); lastScore = 0; nFrames = 0; settle = 0; resetDiag();
    if (state != before) sendStatus();
  }
  else if (settle < SETTLE) {                // just reached running level: let the speed settle first
    settle++; state = SPIN; lastScore = 0;
  }
  else {
    lastScore = score(v);
    votes[vIdx] = lastScore > base.thr; vIdx = (vIdx + 1) % VOTES;
    if (lastScore > base.thr) {                // abnormal: which known fault does it look like? (majority since it started)
      float mt; int d = diagnose(v, &mt); diagCount[d + 1]++;
      int bi = 0; for (int i = 1; i <= N_FAULTS; i++) if (diagCount[i] > diagCount[bi]) bi = i;
      diag = bi - 1; if (d == diag) diagMatch = mt;
    }
    for (int i = 0; i < VOTES; i++) abnormal += votes[i];
    State prev = state;
    if (abnormal >= ALERT_VOTES) state = ALERT;
    else if (state == ALERT && abnormal > CLEAR_VOTES) state = ALERT;   // hold until clearly healthy
    else if (abnormal >= 2) state = WATCH;
    else state = OK_;
    if (state == OK_ && prev != OK_) resetDiag();
    if (state != prev) sendStatus();
  }
  Serial.printf("D,%lu,%s,%.2f,%.2f,%.1f,%d,%s,%.2f\n", (unsigned long)now, STATE_NAME[state], lastScore, base.thr, lastRms, abnormal, diagCode(), diagMatch);
  if (now - lastBle > 2000) { sendStatus(); lastBle = now; }
}
