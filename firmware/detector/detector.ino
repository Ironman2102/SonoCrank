// detector.ino - on-device "learn-your-machine" anomaly detector.
//   1. Hold the button for 1 s (or type c in Serial Monitor) while the machine runs normally.
//   2. The ESP32 records ~30 s of healthy vibration and learns its sound (saved across resets).
//   3. From then on it scores every 0.5 s of sound against what it learned and raises an alert
//      (red LED + buzzer + Bluetooth message) when the sound stays abnormal.
// Board: "ESP32 Dev Module", esp32 core 2.0.17.
// If upload says "Sketch too big": Tools -> Partition Scheme -> "Huge APP (3MB No OTA/1MB SPIFFS)".
// Phone: install "nRF Connect", connect to "EngineSense", open Nordic UART Service,
//        enable notifications on the TX characteristic and show values as text (UTF-8).
// v5: a change must last CONFIRM_S seconds before it becomes an ALERT (WATCH = "observing"),
//     short changes are reported as TRANSIENT; optional Wi-Fi alerts by MQTT + phone push (ntfy).

// ---------------- optional: alerts over Wi-Fi (MQTT + phone notification) ----------------
// USE_WIFI 0 = Bluetooth only (no extra setup).
// USE_WIFI 1 = also send alerts over Wi-Fi (e.g. your phone hotspot WITH mobile data). Then:
//   1. Arduino IDE: Sketch -> Include Library -> Manage Libraries -> install "PubSubClient" (Nick O'Leary)
//   2. Tools -> Partition Scheme -> "Huge APP (3MB No OTA/1MB SPIFFS)"  (Wi-Fi makes the program bigger)
//   3. Fill in WIFI_SSID / WIFI_PASS below
//   4. Phone: install the free "ntfy" app, tap +, subscribe to the topic in NTFY_TOPIC -> alerts arrive as notifications
//   5. Optional MQTT dashboard: any MQTT app (e.g. "IoT MQTT Panel"), broker below, topic TOPIC_BASE/#
#define USE_WIFI    0
#define WIFI_SSID   "your-hotspot-name"
#define WIFI_PASS   "your-hotspot-password"
#define MQTT_HOST   "broker.hivemq.com"        // free public test broker (anyone can read it: no private data is sent)
#define MQTT_PORT   1883
#define TOPIC_BASE  "sonocrank/rvu-g4"          // publishes TOPIC_BASE/status (every 2 s), /alert (retained), /event
#define NTFY_TOPIC  "sonocrank-rvu-g4-alerts"   // must match the topic you subscribe to in the ntfy app

#include <Arduino.h>
#include <Preferences.h>
#include "driver/i2s.h"
#include "driver/adc.h"
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLE2902.h>
#include "dsp.h"                       // on-device MFCC, identical to features.py
#include "faults.h"                    // fault fingerprints (made by gen_faults.py) for the fault suggestion
#include "ae_model.h"                  // speed-aware autoencoder (made by export_ae.py); AE_AVAILABLE 0 = not included
#if USE_WIFI
#if !__has_include(<PubSubClient.h>)
#error "USE_WIFI is 1: install the PubSubClient library (Sketch -> Include Library -> Manage Libraries)"
#endif
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <PubSubClient.h>
#endif

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
#define CONFIRM_S   10        // a change must last about this long (seconds) before it becomes an ALERT
#define CONFIRM_DEC (CONFIRM_S * 2)   // = number of decisions (one every 0.5 s)
#define CONFIRM_FRAC 0.75f    // ... with at least 75 % of the decisions in that time abnormal

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
// ---- speed-aware autoencoder: judges against healthy sound learned over a RANGE of engine speeds ----
static bool useAE = false;                 // 'a' in Serial Monitor switches between autoencoder and z-score
static float aeThr = 0;                    // alert level: trained level, raised if this machine's calibration needs it
static float calFeat[CAL_WINDOWS][2 * N_MFCC];
static float lastAeScore = 0;
static int diagCount[N_FAULTS + 1]; static int diag = -2; static float diagMatch = 0;   // -2 none, -1 unknown, else fault index
static uint8_t votes[VOTES]; static int vIdx = 0;
static bool muted = false;
static uint32_t btnDown = 0; static bool btnFired = false;
static uint32_t lastBle = 0;
static float lastScore = 0, lastRms = 0;
static uint8_t hist[CONFIRM_DEC]; static int hIdx = 0, watchN = 0;   // decisions observed since the change started
static int transients = 0;
const char* faultTitle(const char* code) { return !strcmp(code, "IMB") ? "Imbalance or looseness" : !strcmp(code, "RUB") ? "Rubbing or friction" : "Unusual vibration"; }

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

// ---------------- Wi-Fi alerts: run on their own task so sending never interrupts the sound sampling ----------------
#if USE_WIFI
struct NetMsg { char kind; char text[72]; };        // 's' status, 'a' alert (push + retained), 'c' cleared, 'e' event
static QueueHandle_t netQ = nullptr;
static WiFiClient mqttNet; static PubSubClient mqtt(mqttNet);
static volatile bool wifiUp = false;
void netPost(char kind, const char* text) {
  if (!netQ) return;
  NetMsg m; m.kind = kind; strncpy(m.text, text, sizeof(m.text) - 1); m.text[sizeof(m.text) - 1] = 0;
  xQueueSend(netQ, &m, 0);                          // never wait: drop if the queue is full
}
int ntfyTry(WiFiClient& c, const char* url, const char* title, const char* body, const char* prio, const char* tags) {
  HTTPClient http; http.setTimeout(4000);
  if (!http.begin(c, url)) return -1;
  http.addHeader("Title", title); http.addHeader("Priority", prio); http.addHeader("Tags", tags);
  int code = http.POST((uint8_t*)body, strlen(body)); http.end(); return code;
}
void ntfyPush(const char* title, const char* body, const char* prio, const char* tags) {
  WiFiClient plain; int code = ntfyTry(plain, "http://ntfy.sh/" NTFY_TOPIC, title, body, prio, tags);
  if (code != 200) { WiFiClientSecure sec; sec.setInsecure(); code = ntfyTry(sec, "https://ntfy.sh/" NTFY_TOPIC, title, body, prio, tags); }
  Serial.printf("# phone notification %s (%d)\n", code == 200 ? "sent" : "FAILED", code);
}
void netTask(void*) {
  WiFi.mode(WIFI_STA); WiFi.setAutoReconnect(true); WiFi.begin(WIFI_SSID, WIFI_PASS);
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  uint32_t lastTry = 0; bool wasUp = false;
  char id[32]; snprintf(id, sizeof(id), "sonocrank-%06lx", (unsigned long)(ESP.getEfuseMac() & 0xFFFFFF));
  for (;;) {
    wifiUp = WiFi.status() == WL_CONNECTED;
    if (wifiUp != wasUp) { wasUp = wifiUp; Serial.printf("# wifi %s\n", wifiUp ? "connected" : "disconnected"); }
    if (wifiUp && !mqtt.connected() && millis() - lastTry > 5000) {
      lastTry = millis();
      if (mqtt.connect(id, TOPIC_BASE "/online", 0, true, "0")) { mqtt.publish(TOPIC_BASE "/online", "1", true); Serial.println("# mqtt connected"); }
    }
    if (mqtt.connected()) mqtt.loop();
    NetMsg m;
    if (xQueueReceive(netQ, &m, pdMS_TO_TICKS(100)) == pdTRUE && wifiUp) {
      if (m.kind == 's') { if (mqtt.connected()) mqtt.publish(TOPIC_BASE "/status", m.text); }
      else if (m.kind == 'e') { if (mqtt.connected()) mqtt.publish(TOPIC_BASE "/event", m.text); }
      else if (m.kind == 'a') {
        if (mqtt.connected()) mqtt.publish(TOPIC_BASE "/alert", m.text, true);
        char title[64]; snprintf(title, sizeof(title), "Engine ALERT: %s", faultTitle(strrchr(m.text, ' ') ? strrchr(m.text, ' ') + 1 : ""));
        ntfyPush(title, "Abnormal vibration confirmed. Stop when safe and inspect the engine.", "urgent", "warning,motorcycle");
      }
      else if (m.kind == 'c') {
        if (mqtt.connected()) mqtt.publish(TOPIC_BASE "/alert", "OK", true);
        ntfyPush("Engine back to normal", "Vibration matches the healthy engine again.", "default", "white_check_mark");
      }
    }
  }
}
void setupNet() { netQ = xQueueCreate(8, sizeof(NetMsg)); xTaskCreatePinnedToCore(netTask, "net", 12288, nullptr, 1, nullptr, 0); }
#else
void netPost(char, const char*) {}
void setupNet() {}
#endif

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

#if AE_AVAILABLE
float aeScore(const float* feat) {          // 26 -> 16 -> 6 -> 16 -> 26, ReLU; score = RMS reconstruction error
  float x[AE_IN], h1[AE_H], z[AE_Z], h2[AE_H], e = 0;
  for (int i = 0; i < AE_IN; i++) x[i] = (feat[i] - AE_MU[i]) / AE_SD[i];
  for (int j = 0; j < AE_H; j++) { float a = AE_B1[j]; for (int i = 0; i < AE_IN; i++) a += x[i] * AE_W1[i][j]; h1[j] = a > 0 ? a : 0; }
  for (int j = 0; j < AE_Z; j++) { float a = AE_B2[j]; for (int i = 0; i < AE_H; i++) a += h1[i] * AE_W2[i][j]; z[j] = a > 0 ? a : 0; }
  for (int j = 0; j < AE_H; j++) { float a = AE_B3[j]; for (int i = 0; i < AE_Z; i++) a += z[i] * AE_W3[i][j]; h2[j] = a > 0 ? a : 0; }
  for (int j = 0; j < AE_IN; j++) { float a = AE_B4[j]; for (int i = 0; i < AE_H; i++) a += h2[i] * AE_W4[i][j]; e += (a - x[j]) * (a - x[j]); }
  return sqrtf(e / AE_IN);
}
#else
float aeScore(const float* feat) { return 0; }
#endif
bool aeOn() { return AE_AVAILABLE && useAE && aeThr > 0; }
float curThr() { return aeOn() ? aeThr : base.thr; }

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
void sendStatus() {                    // e.g. "OK 0.62/1.40", "WATCH 2.10/1.40 IMB T3/8", "ALERT 2.30/1.40 RUB"
  char m[40]; int n;
  if (diag != -2 && (state == WATCH || state == ALERT)) n = snprintf(m, sizeof(m), "%s %.2f/%.2f %s", STATE_NAME[state], lastScore, curThr(), diagCode());
  else n = snprintf(m, sizeof(m), "%s %.2f/%.2f", STATE_NAME[state], lastScore, curThr());
  if (state == WATCH) snprintf(m + n, sizeof(m) - n, " T%d/%d", watchN / 2, CONFIRM_S);   // seconds observed so far
  bleSend(m);
  netPost('s', m);
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
  base.runRms = r[calN / 5];                 // 20th percentile: the quietest normal running (idle) still counts as running
#if AE_AVAILABLE
  float amx = 0; for (int i = 0; i < calN; i++) amx = fmaxf(amx, aeScore(calFeat[i]));
  aeThr = fmaxf(AE_THR, 1.5f * amx);          // never below the trained level; raised if this machine/mounting needs it
  prefs.putFloat("aethr", aeThr);
  Serial.printf("# autoencoder: worst healthy score %.3f, alert level %.3f (trained %.3f)\n", amx, aeThr, AE_THR);
#endif
  base.magic = MAGIC;
  prefs.putBytes("base", &base, sizeof(base));
  state = OK_; settle = SETTLE;          // the machine is already running steadily
  memset(votes, 0, sizeof(votes));
  Serial.printf("# calibration done: %d windows, worst healthy score %.2f, alert threshold %.2f, running level %.1f\n", calN, mx, base.thr, base.runRms);
  Serial.printf("# detector: %s\n", aeOn() ? "speed-aware autoencoder" : "z-score");
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
    else if (ch == 'a') {
      if (!AE_AVAILABLE) Serial.println("# no autoencoder in this build (ae_model.h missing)");
      else { useAE = !useAE; prefs.putBool("useae", useAE); memset(votes, 0, sizeof(votes)); resetDiag();
             Serial.printf("# detector: %s%s\n", useAE ? "speed-aware autoencoder" : "z-score", useAE && aeThr <= 0 ? " (after the next calibration)" : ""); }
    }
    else if (ch == '?') Serial.println("# c = calibrate, x = clear calibration, o = set machine-off level, m = mute buzzer, a = autoencoder on/off");
    else if (ch == 'n') { if (!USE_WIFI) Serial.println("# Wi-Fi alerts are off (USE_WIFI 0)"); else { netPost('a', "ALERT TEST IMB"); Serial.println("# test phone notification queued"); } }
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
  useAE = AE_AVAILABLE && prefs.getBool("useae", true);
  aeThr = (AE_AVAILABLE && state != NOCAL) ? prefs.getFloat("aethr", AE_THR) : 0;
  setupAdc();
  setupBle();
  setupNet();
  Serial.printf("# alert after a change lasts %d s; Wi-Fi alerts %s\n", CONFIRM_S, USE_WIFI ? "ON (type n to send a test notification)" : "off");
  Serial.printf("# EngineSense ready. %s. Type ? for commands.\n",
                state == NOCAL ? "Not calibrated: hold the button 1 s" : "Calibration loaded");
  Serial.printf("# detector: %s\n", aeOn() ? "speed-aware autoencoder" : (AE_AVAILABLE && useAE ? "speed-aware autoencoder (after calibration)" : "z-score"));
  Serial.println("# columns: D,ms,state,score,threshold,rms,abnormal_votes,fault,match,watch_s");
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
  float feat[2 * N_MFCC];                    // autoencoder input: mean and spread over the window of each MFCC
  for (int c = 0; c < N_MFCC; c++) {
    double q = 0; for (int f = 0; f < WIN_FRAMES; f++) { double d = mf[f][c] - v[c]; q += d * d; }
    feat[c] = v[c]; feat[N_MFCC + c] = sqrt(q / WIN_FRAMES);
  }
  double mean = sumV / nS; lastRms = sqrt(fmax(0.0, sumSq / nS - mean * mean));
  sumV = sumSq = 0; nS = 0;

  if (state == CALIB) {
    if (lastRms >= offRms) { calRms[calN] = lastRms; memcpy(calFeat[calN], feat, sizeof(feat)); memcpy(calVec[calN++], v, sizeof(v)); }   // learn only while running
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
    float zs = score(v);
    lastScore = aeOn() ? aeScore(feat) : zs;
    bool high = lastScore > curThr();
    votes[vIdx] = high; vIdx = (vIdx + 1) % VOTES;
    if (lastScore > curThr()) {                // abnormal: which known fault does it look like? (majority since it started)
      float mt; int d = diagnose(v, &mt); diagCount[d + 1]++;
      int bi = 0; for (int i = 1; i <= N_FAULTS; i++) if (diagCount[i] > diagCount[bi]) bi = i;
      diag = bi - 1; if (d == diag) diagMatch = mt;
    }
    for (int i = 0; i < VOTES; i++) abnormal += votes[i];
    State prev = state;
    if (state == ALERT) {                      // confirmed fault: hold until clearly healthy again
      if (abnormal <= CLEAR_VOTES) state = OK_;
    } else if (state == WATCH) {               // observing a change: is it lasting, or just a bump?
      hist[hIdx] = high; hIdx = (hIdx + 1) % CONFIRM_DEC; watchN++;
      int recent = 0; for (int i = 0; i < CONFIRM_DEC; i++) recent += hist[i];
      if (abnormal <= CLEAR_VOTES) {           // it went away: short change, not a fault
        state = OK_; transients++;
        char e[48]; snprintf(e, sizeof(e), "TRANSIENT %.1fs", watchN * 0.5f);
        Serial.printf("# short change (%.1f s), not an alert\n", watchN * 0.5f); bleSend(e); netPost('e', e);
      } else if (watchN >= CONFIRM_DEC && recent >= (int)(CONFIRM_FRAC * CONFIRM_DEC + 0.5f)) state = ALERT;
    } else if (abnormal >= 2) {                // something changed: start observing
      state = WATCH; memset(hist, 0, sizeof(hist)); hIdx = 0; watchN = 0;
      hist[0] = high; hIdx = 1; watchN = 1;
    } else state = OK_;
    if (state == OK_ && prev != OK_) resetDiag();
    if (state == ALERT && prev != ALERT) { char a[40]; snprintf(a, sizeof(a), "ALERT %.2f/%.2f %s", lastScore, curThr(), diagCode()); netPost('a', a); }
    if (prev == ALERT && state == OK_) netPost('c', "OK");
    if (state != prev || state == WATCH) sendStatus();   // while observing, update the countdown every 0.5 s
  }
  Serial.printf("D,%lu,%s,%.3f,%.3f,%.1f,%d,%s,%.2f,%.1f\n", (unsigned long)now, STATE_NAME[state], lastScore, curThr(), lastRms, abnormal, diagCode(), diagMatch, state == WATCH ? watchN * 0.5f : 0.0f);
  if (now - lastBle > 2000) { sendStatus(); lastBle = now; }
}
