// vibe_check.ino - is the piezo picking up the fan?
// Samples GPIO34 at 10 kHz for 0.4 s (long enough to see slow fan vibration),
// then prints the resting level, the vibration size (RMS) and peak-to-peak.
// Board: "ESP32 Dev Module". Serial Monitor at 115200.
const int PIN_ADC = 34;
const int N = 4000;            // 4000 samples x 100 us = 0.4 s
static uint16_t buf[N];

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetPinAttenuation(PIN_ADC, ADC_11db);
  Serial.println("vibe_check: fan OFF first, then ON. Compare the RMS numbers.");
}

void loop() {
  uint32_t t = micros();
  for (int i = 0; i < N; i++) {          // steady 10 kHz sampling
    while ((uint32_t)(micros() - t) < 100) {}
    t += 100;
    buf[i] = analogRead(PIN_ADC);
  }
  double sum = 0; int lo = 4095, hi = 0;
  for (int i = 0; i < N; i++) { sum += buf[i]; if (buf[i] < lo) lo = buf[i]; if (buf[i] > hi) hi = buf[i]; }
  double mean = sum / N, var = 0;
  for (int i = 0; i < N; i++) { double d = buf[i] - mean; var += d * d; }
  double rms = sqrt(var / N);
  Serial.printf("mean %4.0f  RMS %6.1f  p-p %4d  min %4d  max %4d  ", mean, rms, hi - lo, lo, hi);
  int bars = (int)(rms / 2); if (bars > 50) bars = 50;     // quick visual
  for (int i = 0; i < bars; i++) Serial.print('#');
  Serial.println();
}
