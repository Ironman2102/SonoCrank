// logger.ino - Phase 2 raw-audio logger. Arduino-ESP32 core 2.0.x, board "ESP32 Dev Module".
// Piezo front-end output -> GPIO34 (ADC1_CH6). Streams 12-bit counts to the PC at 16 kHz.
#include <Arduino.h>
#include "driver/i2s.h"
#include "driver/adc.h"
#define SR  16000
#define BLK 512
static uint16_t buf[BLK];
static uint8_t seq = 0;

void setup() {
  Serial.begin(921600);
  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_ADC_BUILT_IN);
  cfg.sample_rate = SR;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = 8;            // 8 x 512 samples = 256 ms of safety margin
  cfg.dma_buf_len = BLK;
  cfg.use_apll = false;
  i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL);
  adc1_config_width(ADC_WIDTH_BIT_12);
  adc1_config_channel_atten(ADC1_CHANNEL_6, ADC_ATTEN_DB_11);   // ~0.1-3.1 V usable range
  i2s_set_adc_mode(ADC_UNIT_1, ADC1_CHANNEL_6);
  i2s_adc_enable(I2S_NUM_0);
}

void loop() {
  size_t got = 0;
  i2s_read(I2S_NUM_0, buf, sizeof(buf), &got, portMAX_DELAY);
  if (got != sizeof(buf)) return;
  for (int i = 0; i < BLK; i++) buf[i] &= 0x0FFF;   // upper 4 bits carry channel info
  Serial.write(0xA5); Serial.write(0x5A); Serial.write(seq++);
  Serial.write((uint8_t*)buf, sizeof(buf));
}
