// hello.ino - Step 2: first upload. Checks the board, the USB link and the chip type.
// Board: "ESP32 Dev Module". Serial Monitor at 115200 baud.

void setup() {
  Serial.begin(115200);
  delay(1000);
  pinMode(2, OUTPUT);   // blue LED on most 30-pin boards
  Serial.println();
  Serial.printf("Chip: %s, revision %d, %d cores\n",
                ESP.getChipModel(), (int)ESP.getChipRevision(), (int)ESP.getChipCores());
  Serial.printf("Flash: %u MB, free RAM: %u KB\n",
                (unsigned)(ESP.getFlashChipSize() / (1024 * 1024)),
                (unsigned)(ESP.getFreeHeap() / 1024));
}

void loop() {
  static unsigned n = 0;
  digitalWrite(2, HIGH); delay(500);
  digitalWrite(2, LOW);  delay(500);
  Serial.printf("ESP32 alive %u\n", ++n);
}
