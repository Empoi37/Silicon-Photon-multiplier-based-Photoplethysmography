/*
 * ESP32 PPG-SiPM — full firmware (GUI + OLED + BLE)
 *
 * Open this sketch folder in Arduino IDE: firmware_esp32_full/
 * For noise-debug builds use ../firmware_esp32/firmware_esp32.ino (barebones).
 *
 * I2C (SDA=21, SCL=22):
 *   0x18  LIS2DH12   accelerometer
 *   0x2E  MCP4531    TIA gain
 *   0x2F  MCP4018    SiPM bias (BOOST)
 *   0x48  ADS1115    AIN0 = /AMP_OUT (TIA)
 *   0x3C  OLED SSD1306 128x32 (alt 0x3D)
 *
 * GPIO:
 *   18  HV_EN     HIGH = HV on
 *   16  LED1      PWM
 *   17  LED2      PWM
 *   25  DAC_ALC   held at 0 (3.3 V TIA ref)
 *
 * USB serial @115200: one CSV line per sample (250 Hz), full rate for GUI.
 * BLE Nordic UART: batched packets (nRF Connect) to keep the loop fast.
 *
 * Libraries: Adafruit ADS1X15, Adafruit SSD1306, Adafruit GFX, SparkFun LIS2DH12
 */

#include <Wire.h>
#include <math.h>
#include <string.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_ADS1X15.h>
#include <SparkFun_LIS2DH12.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ---- Pins / bus ----
constexpr uint8_t  PIN_SDA = 21, PIN_SCL = 22;
constexpr uint32_t I2C_FREQ = 400000;
constexpr uint8_t  ADDR_LIS2DH12 = 0x18, ADDR_MCP4531 = 0x2E,
                   ADDR_MCP4018 = 0x2F, ADDR_ADS1115 = 0x48;
constexpr uint8_t  ADDR_OLED_PRIMARY = 0x3C, ADDR_OLED_ALT = 0x3D;
constexpr uint8_t  PIN_HV_EN = 18, PIN_LED1 = 16, PIN_LED2 = 17, PIN_DAC_ALC = 25;

// ---- OLED ----
constexpr uint8_t  OLED_W = 128, OLED_H = 32;
constexpr int8_t   OLED_RESET = -1;
constexpr uint32_t OLED_REFRESH_US = 500000;  // 2 Hz (was 4 Hz — less I2C on shared bus)

// ---- BLE (Nordic UART) ----
static constexpr char BLE_NAME[] = "PPG-SiPM";
static constexpr char NUS_SERVICE_UUID[] = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E";
static constexpr char NUS_TX_UUID[]      = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E";
static constexpr char NUS_RX_UUID[]      = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E";
constexpr uint8_t  BLE_BATCH_SAMPLES = 20;   // 100 ms between flushes @ 250 Hz
constexpr uint32_t BLE_FLUSH_MAX_US = 100000;
constexpr uint32_t BLE_MIN_GAP_US = 3000;    // no RF burst within 3 ms of ADC read

// ---- Acquisition ----
constexpr uint32_t PWM_FREQ = 5000;
constexpr uint8_t  PWM_RES_BITS = 8;
constexpr uint32_t SAMPLE_RATE_HZ = 250;
constexpr uint32_t SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE_HZ;
constexpr int16_t  ADS_SINGLE_ENDED_MAX = 32767;

// ---- Default front-end (GUI screenshot 2026-07-11) ----
constexpr uint8_t  DEFAULT_ADS_GAIN = 0;
constexpr uint8_t  DEFAULT_BOOST = 63;
constexpr bool     DEFAULT_HV_ENABLED = true;

static const adsGain_t ADS_GAIN_TABLE[] = {
    GAIN_TWOTHIRDS, GAIN_ONE, GAIN_TWO, GAIN_FOUR, GAIN_EIGHT, GAIN_SIXTEEN};

// ---- Devices ----
Adafruit_SSD1306 display(OLED_W, OLED_H, &Wire, OLED_RESET);
Adafruit_ADS1115 ads;
SPARKFUN_LIS2DH12 lis;
bool adsReady = false;
bool lisReady = false;
bool oledReady = false;

// ---- Front-end state ----
uint8_t ledLevel1  = 115;
uint8_t ledLevel2  = 118;
uint8_t tiaGain    = 34;
uint8_t boost      = DEFAULT_BOOST;
uint8_t dacOffset  = 0;
uint8_t adsGainIdx = DEFAULT_ADS_GAIN;
bool    hvEnabled  = DEFAULT_HV_ENABLED;

// ---- Serial / BLE command buffers ----
char    cmdBuf[64];
uint8_t cmdLen = 0;
char    bleCmdBuf[64];
uint8_t bleCmdLen = 0;

BLECharacteristic *bleTxChar = nullptr;
bool bleConnected = false;
bool bleNotifyOn = false;
bool bleHeaderPending = true;

// ---- BLE batch TX ----
struct SamplePkt {
  int16_t ppg;
  int16_t ax;
  int16_t ay;
  int16_t az;
};
static SamplePkt bleBatch[BLE_BATCH_SAMPLES];
static uint8_t bleBatchCount = 0;
static uint32_t bleLastFlushUs = 0;
static uint32_t lastAdcSampleUs = 0;
static bool bleFlushPending = false;
static char blePacketBuf[480];

// ---------------------------------------------------------------------------
// Digipots / actuators
// ---------------------------------------------------------------------------
static void writeMCP4531(uint8_t value) {
  Wire.beginTransmission(ADDR_MCP4531);
  Wire.write(0x00);
  Wire.write(value);
  Wire.endTransmission();
}

static void writeMCP4018(uint8_t value) {
  Wire.beginTransmission(ADDR_MCP4018);
  Wire.write(value & 0x7F);
  Wire.endTransmission();
}

static void setTiaGain(uint8_t v)   { tiaGain = v;      writeMCP4531(v); }
static void setBoost(uint8_t v)     { boost = v & 0x7F; writeMCP4018(boost); }
static void setDacOffset(uint8_t v) {
  (void)v;
  dacOffset = 0;
  dacWrite(PIN_DAC_ALC, 0);
}
static void setHighVoltage(bool en) {
  hvEnabled = en;
  digitalWrite(PIN_HV_EN, en ? HIGH : LOW);
}
static void maintainHighVoltage() {
  if (hvEnabled) digitalWrite(PIN_HV_EN, HIGH);
}
static void setLeds(uint8_t l1, uint8_t l2) {
  ledLevel1 = l1;
  ledLevel2 = l2;
  ledcWrite(PIN_LED1, l1);
  ledcWrite(PIN_LED2, l2);
}

static void setAdsGain(uint8_t idx) {
  if (!adsReady) return;
  adsGainIdx = constrain(idx, 0, 5);
  ads.setGain(ADS_GAIN_TABLE[adsGainIdx]);
  ads.startADCReading(ADS1X15_REG_CONFIG_MUX_SINGLE_0, /*continuous=*/true);
}

static int16_t readAdc() {
  return adsReady ? ads.getLastConversionResults() : 0;
}

static int16_t ppgFromAdc(int16_t raw) {
  int32_t r = (int32_t)raw;
  if (r < 0) r = 0;
  if (r > ADS_SINGLE_ENDED_MAX) r = ADS_SINGLE_ENDED_MAX;
  return (int16_t)(ADS_SINGLE_ENDED_MAX - r);
}

// ---------------------------------------------------------------------------
// On-device BPM (lightweight)
// ---------------------------------------------------------------------------
struct BpmEstimator {
  static constexpr float    LP_ALPHA = 0.125f;
  static constexpr float    HP_ALPHA = 0.0125f;
  static constexpr float    THRESH_DECAY = 0.997f;
  static constexpr float    THRESH_FRAC = 0.50f;
  static constexpr float    THRESH_MIN = 2.0f;
  static constexpr uint32_t REFRACTORY_US = 280000;
  static constexpr uint32_t MIN_IBI_US = 300000;
  static constexpr uint32_t MAX_IBI_US = 1500000;
  static constexpr uint32_t STALE_US = 2500000;
  static constexpr uint8_t  IBI_HIST = 5;

  float lp = 0, hp = 0, v0 = 0, v1 = 0, peakThresh = THRESH_MIN;
  uint32_t lastPeakUs = 0;
  uint32_t ibiUs[IBI_HIST] = {};
  uint8_t ibiCount = 0, ibiIdx = 0;
  bool locked = false;
  uint16_t bpmShown = 0;

  void reset() {
    lp = hp = v0 = v1 = 0;
    peakThresh = THRESH_MIN;
    lastPeakUs = 0;
    ibiCount = ibiIdx = 0;
    locked = false;
    bpmShown = 0;
    memset(ibiUs, 0, sizeof(ibiUs));
  }

  static uint32_t medianIbi(const uint32_t *vals, uint8_t n) {
    uint32_t tmp[IBI_HIST];
    for (uint8_t i = 0; i < n; ++i) tmp[i] = vals[i];
    for (uint8_t i = 1; i < n; ++i) {
      uint32_t v = tmp[i];
      int j = (int)i - 1;
      while (j >= 0 && tmp[j] > v) { tmp[j + 1] = tmp[j]; --j; }
      tmp[j + 1] = v;
    }
    return tmp[n / 2];
  }

  void push(int16_t sample, uint32_t nowUs) {
    lp += LP_ALPHA * ((float)sample - lp);
    hp += HP_ALPHA * (lp - hp);
    float filtered = lp - hp;

    if (v0 < v1 && v1 > filtered && v1 > peakThresh) {
      if (lastPeakUs == 0 || (nowUs - lastPeakUs) >= REFRACTORY_US) {
        if (lastPeakUs != 0) {
          uint32_t ibi = nowUs - lastPeakUs;
          if (ibi >= MIN_IBI_US && ibi <= MAX_IBI_US) {
            ibiUs[ibiIdx] = ibi;
            ibiIdx = (ibiIdx + 1) % IBI_HIST;
            if (ibiCount < IBI_HIST) ++ibiCount;
            locked = true;
          }
        }
        lastPeakUs = nowUs;
        peakThresh = fmaxf(THRESH_MIN, v1 * THRESH_FRAC);
      }
    }
    peakThresh *= THRESH_DECAY;
    if (peakThresh < THRESH_MIN) peakThresh = THRESH_MIN;

    if (locked && lastPeakUs && (nowUs - lastPeakUs) > STALE_US) {
      locked = false;
      ibiCount = 0;
      bpmShown = 0;
      peakThresh = THRESH_MIN;
    }
    v0 = v1;
    v1 = filtered;
  }

  uint16_t bpmInstant() const {
    if (!locked || ibiCount < 2) return 0;
    uint32_t med = medianIbi(ibiUs, ibiCount);
    if (!med) return 0;
    float bpm = 60.0e6f / (float)med;
    if (bpm < 40.f || bpm > 200.f) return 0;
    return (uint16_t)(bpm + 0.5f);
  }

  uint16_t bpmSmoothed() {
    uint16_t instant = bpmInstant();
    if (!instant) {
      if (!locked) bpmShown = 0;
      return bpmShown;
    }
    if (!bpmShown) bpmShown = instant;
    else bpmShown = (uint16_t)(0.65f * bpmShown + 0.35f * instant + 0.5f);
    return bpmShown;
  }

  bool isValid() const { return locked && ibiCount >= 2; }
};

static BpmEstimator bpmEst;
static int16_t g_lastPpg = 0;

// ---------------------------------------------------------------------------
// OLED diagnostic
// ---------------------------------------------------------------------------
static bool initOled() {
  if (display.begin(SSD1306_SWITCHCAPVCC, ADDR_OLED_PRIMARY)) return true;
  if (display.begin(SSD1306_SWITCHCAPVCC, ADDR_OLED_ALT)) return true;
  return false;
}

static void drawOled(uint16_t bpm, bool valid, int16_t lastPpg) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);

  display.setCursor(0, 0);
  if (valid && bpm >= 40 && bpm <= 200) display.printf("HR %3u BPM", bpm);
  else display.print("HR  -- BPM");
  display.setCursor(88, 0);
  display.print(hvEnabled ? "HV ON" : "HV off");

  display.setCursor(0, 8);
  display.printf("G%-3u B%-3u ADS%u", tiaGain, boost, adsGainIdx);
  display.setCursor(100, 8);
  if (bleConnected && bleNotifyOn) display.print("BT");
  else if (bleConnected) display.print("bt");

  display.setCursor(0, 16);
  display.printf("L1:%-3u L2:%-3u", ledLevel1, ledLevel2);

  display.setCursor(0, 24);
  display.printf("PPG:%-5d", lastPpg);
  if (bleNotifyOn) {
    display.setCursor(88, 24);
    display.print("LOG");
  }
  display.display();
}

// ---------------------------------------------------------------------------
// BLE
// ---------------------------------------------------------------------------
static void dispatchCommand(char *line);

class BleServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *server) override {
    bleConnected = true;
    bleNotifyOn = false;
    bleHeaderPending = true;
    bleBatchCount = 0;
    BLEDevice::setMTU(517);
  }
  void onDisconnect(BLEServer *server) override {
    bleConnected = false;
    bleNotifyOn = false;
    bleHeaderPending = true;
    bleBatchCount = 0;
    BLEDevice::startAdvertising();
  }
};

class BleNotifyCallbacks : public BLEDescriptorCallbacks {
  void onWrite(BLEDescriptor *descriptor) override {
    uint8_t *val = descriptor->getValue();
    bleNotifyOn = (val[0] & 0x01) != 0;
    if (bleNotifyOn) {
      bleHeaderPending = true;
      bleBatchCount = 0;
    }
  }
};

class BleRxCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *characteristic) override {
    String rx = characteristic->getValue();
    for (unsigned int i = 0; i < rx.length(); ++i) {
      char c = rx.charAt(i);
      if (c == '\n' || c == '\r') {
        if (bleCmdLen > 0) {
          bleCmdBuf[bleCmdLen] = '\0';
          dispatchCommand(bleCmdBuf);
          bleCmdLen = 0;
        }
      } else if (bleCmdLen < sizeof(bleCmdBuf) - 1) {
        bleCmdBuf[bleCmdLen++] = c;
      } else {
        bleCmdLen = 0;
      }
    }
  }
};

static void initBle() {
  BLEDevice::init(BLE_NAME);
  BLEDevice::setPower(ESP_PWR_LVL_N0);  // lower TX power -> less analog supply coupling
  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(new BleServerCallbacks());

  BLEService *service = server->createService(NUS_SERVICE_UUID);
  bleTxChar = service->createCharacteristic(NUS_TX_UUID, BLECharacteristic::PROPERTY_NOTIFY);
  BLE2902 *ccc = new BLE2902();
  ccc->setCallbacks(new BleNotifyCallbacks());
  bleTxChar->addDescriptor(ccc);

  BLECharacteristic *rx = service->createCharacteristic(
      NUS_RX_UUID,
      BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR);
  rx->setCallbacks(new BleRxCallbacks());

  service->start();
  BLEAdvertising *adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(NUS_SERVICE_UUID);
  adv->setScanResponse(true);
  BLEDevice::startAdvertising();
}

static void bleFlushBatch(bool forcePartial) {
  if (!bleConnected || !bleNotifyOn || bleTxChar == nullptr) return;
  if (bleBatchCount == 0) return;
  if (!forcePartial && bleBatchCount < BLE_BATCH_SAMPLES) return;

  // Avoid BLE RF burst right next to an ADC sample (supply bounce -> HF ADC noise).
  uint32_t now = micros();
  if (lastAdcSampleUs != 0 && (uint32_t)(now - lastAdcSampleUs) < BLE_MIN_GAP_US) {
    bleFlushPending = true;
    return;
  }

  if (bleHeaderPending) {
    bleHeaderPending = false;
    const char *hdr = "# ppg,ppg,ax,ay,az batch\n";
    bleTxChar->setValue((uint8_t *)hdr, strlen(hdr));
    bleTxChar->notify();
  }

  char *p = blePacketBuf;
  size_t remain = sizeof(blePacketBuf);
  for (uint8_t i = 0; i < bleBatchCount; ++i) {
    int n = snprintf(p, remain, "%d,%d,%d,%d,%d\n",
                     bleBatch[i].ppg, bleBatch[i].ppg,
                     bleBatch[i].ax, bleBatch[i].ay, bleBatch[i].az);
    if (n <= 0 || (size_t)n >= remain) break;
    p += n;
    remain -= (size_t)n;
  }
  size_t len = (size_t)(p - blePacketBuf);
  if (len == 0) return;

  bleTxChar->setValue((uint8_t *)blePacketBuf, len);
  bleTxChar->notify();
  bleBatchCount = 0;
  bleLastFlushUs = micros();
  bleFlushPending = false;
}

static void bleQueueSample(int16_t ppg, int16_t ax, int16_t ay, int16_t az, uint32_t nowUs) {
  if (!bleConnected || !bleNotifyOn) return;

  bleBatch[bleBatchCount].ppg = ppg;
  bleBatch[bleBatchCount].ax = ax;
  bleBatch[bleBatchCount].ay = ay;
  bleBatch[bleBatchCount].az = az;
  ++bleBatchCount;

  if (bleBatchCount >= BLE_BATCH_SAMPLES) bleFlushPending = true;
  else if (bleLastFlushUs == 0) bleLastFlushUs = nowUs;
  else if ((uint32_t)(nowUs - bleLastFlushUs) >= BLE_FLUSH_MAX_US) bleFlushPending = true;
}

// Run BLE flush + OLED outside the ADC sample window.
static void serviceBackground(uint32_t nowUs) {
  if (bleFlushPending) bleFlushBatch(bleBatchCount < BLE_BATCH_SAMPLES);

  static uint32_t lastOledUs = 0;
  if (!oledReady) return;
  if ((uint32_t)(nowUs - lastOledUs) < OLED_REFRESH_US) return;
  if (lastAdcSampleUs != 0 && (uint32_t)(nowUs - lastAdcSampleUs) < BLE_MIN_GAP_US) return;

  lastOledUs = nowUs;
  drawOled(bpmEst.bpmSmoothed(), bpmEst.isValid(), g_lastPpg);
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------
static void printStatus() {
  Serial.print("# PONG gain=");   Serial.print(tiaGain);
  Serial.print(" boost=");        Serial.print(boost);
  Serial.print(" dac=");          Serial.print(dacOffset);
  Serial.print(" adsgain=");      Serial.print(adsGainIdx);
  Serial.print(" led1=");         Serial.print(ledLevel1);
  Serial.print(" led2=");         Serial.print(ledLevel2);
  Serial.print(" hv=");           Serial.println(hvEnabled ? 1 : 0);
}

static void dispatchCommand(char *line) {
  if (line[0] == '\0' || line[0] == '#') return;

  char *sep = strchr(line, ':');
  long value = 0;
  bool hasValue = false;
  if (sep) {
    *sep = '\0';
    value = atol(sep + 1);
    hasValue = true;
  }
  for (char *p = line; *p; ++p) *p = toupper(*p);

  if      (!strcmp(line, "GAIN")    && hasValue) setTiaGain(constrain(value, 0, 255));
  else if (!strcmp(line, "BOOST")   && hasValue) setBoost(constrain(value, 0, 127));
  else if (!strcmp(line, "DAC")     && hasValue) setDacOffset(0);
  else if (!strcmp(line, "ADSGAIN") && hasValue) setAdsGain(constrain(value, 0, 5));
  else if (!strcmp(line, "LED1")    && hasValue) setLeds(constrain(value, 0, 255), ledLevel2);
  else if (!strcmp(line, "LED2")    && hasValue) setLeds(ledLevel1, constrain(value, 0, 255));
  else if (!strcmp(line, "HVEN")    && hasValue) {
    setHighVoltage(value != 0);
    Serial.print("# HV ");
    Serial.println(value != 0 ? "ON" : "OFF");
  }
  else if (!strcmp(line, "DCSERVO") && hasValue) setDacOffset(0);
  else if (!strcmp(line, "PING")) printStatus();
  else { Serial.print("# ERR unknown cmd: "); Serial.println(line); }
}

static void pollSerial() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (cmdLen > 0) {
        cmdBuf[cmdLen] = '\0';
        dispatchCommand(cmdBuf);
        cmdLen = 0;
      }
    } else if (cmdLen < sizeof(cmdBuf) - 1) {
      cmdBuf[cmdLen++] = c;
    } else {
      cmdLen = 0;
    }
  }
}

// ---------------------------------------------------------------------------
// Setup / loop
// ---------------------------------------------------------------------------
void setup() {
  pinMode(PIN_HV_EN, OUTPUT);
  digitalWrite(PIN_HV_EN, LOW);

  Serial.begin(115200);
  delay(200);

  Wire.begin(PIN_SDA, PIN_SCL, I2C_FREQ);
  ledcAttach(PIN_LED1, PWM_FREQ, PWM_RES_BITS);
  ledcAttach(PIN_LED2, PWM_FREQ, PWM_RES_BITS);

  setTiaGain(tiaGain);
  setBoost(boost);
  delay(50);
  setDacOffset(0);
  setLeds(ledLevel1, ledLevel2);
  setHighVoltage(DEFAULT_HV_ENABLED);

  if (ads.begin(ADDR_ADS1115)) {
    adsReady = true;
    ads.setDataRate(RATE_ADS1115_860SPS);
    setAdsGain(DEFAULT_ADS_GAIN);
  } else {
    Serial.println("# ERR: ADS1115 not found");
  }

  if (lis.begin(ADDR_LIS2DH12, Wire)) {
    lisReady = true;
    lis.setScale(2);
    lis.setDataRate(LIS2DH12_ODR_400Hz);
  } else {
    Serial.println("# ERR: LIS2DH12 not found");
  }

  if (initOled()) {
    oledReady = true;
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 10);
    display.println("PPG SiPM full");
    display.display();
    delay(300);
    drawOled(0, false, 0);
    Serial.println("# OLED ready");
  } else {
    Serial.println("# WARN: OLED not found");
  }

  initBle();
  bpmEst.reset();
  maintainHighVoltage();

  Serial.print("# PPG-SiPM FULL ready ");
  Serial.print(SAMPLE_RATE_HZ);
  Serial.print(" Hz HV ");
  Serial.print(hvEnabled ? "ON" : "off");
  Serial.print(" BLE batch=");
  Serial.println(BLE_BATCH_SAMPLES);
}

void loop() {
  static uint32_t nextSampleUs = 0;
  static uint32_t lastHvMaintainUs = 0;
  static int16_t ax = 0, ay = 0, az = 0;

  uint32_t now = micros();

  // Background work only between ADC sample ticks.
  if ((int32_t)(now - nextSampleUs) < 0) {
    serviceBackground(now);
    pollSerial();
    return;
  }

  nextSampleUs = now + SAMPLE_PERIOD_US;

  // ---- Critical path: ADC first, minimal latency ----
  int16_t ppg = ppgFromAdc(readAdc());
  lastAdcSampleUs = micros();
  g_lastPpg = ppg;

  if (lisReady && lis.available()) {
    ax = lis.getX();
    ay = lis.getY();
    az = lis.getZ();
  }

  bpmEst.push(ppg, lastAdcSampleUs);

  Serial.printf("%d,%d,%d,%d,%d\n", ppg, ppg, ax, ay, az);
  bleQueueSample(ppg, ax, ay, az, lastAdcSampleUs);

  if ((uint32_t)(lastAdcSampleUs - lastHvMaintainUs) >= 1000000UL) {
    lastHvMaintainUs = lastAdcSampleUs;
    maintainHighVoltage();
  }

  pollSerial();
  serviceBackground(micros());
}
