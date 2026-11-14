// bringup_test.ino - checks the breadboard one part at a time.
// Board: "ESP32 Dev Module". Open Serial Monitor at 115200 baud.
const int PIN_ADC = 34;   // filtered piezo signal (blue wire from i35)
const int PIN_BUZ = 25;   // BC548 base through 1 kOhm
const int PIN_RED = 26;   // red LED (skip if you don't have one)
const int PIN_YEL = 27;   // yellow LED
const int PIN_BTN = 14;   // push button to GND

void setup() {
  Serial.begin(115200);
  pinMode(PIN_BUZ, OUTPUT);
  pinMode(PIN_RED, OUTPUT);
  pinMode(PIN_YEL, OUTPUT);
  pinMode(PIN_BTN, INPUT_PULLUP);
  analogReadResolution(12);                     // 0..4095
  analogSetPinAttenuation(PIN_ADC, ADC_11db);   // about 0..3.1 V range

  // power-on self test: yellow, red, beep
  digitalWrite(PIN_YEL, HIGH); delay(300); digitalWrite(PIN_YEL, LOW);
  digitalWrite(PIN_RED, HIGH); delay(300); digitalWrite(PIN_RED, LOW);
  digitalWrite(PIN_BUZ, HIGH); delay(150); digitalWrite(PIN_BUZ, LOW);
  Serial.println("self test done: yellow, red, beep");
}

void loop() {
  // 400 quick reads: mean = resting level, p-p = size of the vibration signal
  int lo = 4095, hi = 0;
  long sum = 0;
  for (int i = 0; i < 400; i++) {
    int v = analogRead(PIN_ADC);
    sum += v;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  bool pressed = (digitalRead(PIN_BTN) == LOW);
  digitalWrite(PIN_YEL, pressed ? HIGH : LOW);
  if (hi - lo > 600) {                 // a hard tap on the piezo
    digitalWrite(PIN_BUZ, HIGH); delay(40); digitalWrite(PIN_BUZ, LOW);
  }
  Serial.printf("ADC mean %4ld  p-p %4d  min %4d  max %4d  button %s\n",
                sum / 400, hi - lo, lo, hi, pressed ? "PRESSED" : "-");
  delay(150);
}
