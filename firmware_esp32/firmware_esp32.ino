/*
 * ESP32 PPG-SiPM — barebones firmware (Python GUI only)
 *
 * Streams CSV over USB serial @115200. No BLE, no OLED, no on-device BPM.
 * Use this build to debug analog front-end noise without extra I2C/BLE load.
 *
 * I2C (SDA=21, SCL=22):
 *   0x18  LIS2DH12   accelerometer
 *   0x2E  MCP4531    TIA gain
 *   0x2F  MCP4018    SiPM bias (BOOST)
 *   0x48  ADS1115    AIN0 = /AMP_OUT (TIA)
 *
 * GPIO:
 *   18  HV_EN     HIGH = HV on (LOW = safe)
 *   16  LED1      PWM
 *   17  LED2      PWM
 *   25  DAC_ALC   held at 0 (3.3 V TIA reference; servo disabled)
 *
 * Host -> device:  CMD:VALUE
 *   GAIN BOOST DAC ADSGAIN LED1 LED2 HVEN DCSERVO PING
 * Device -> host:  ppg,ppg,ax,ay,az
 */

#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <SparkFun_LIS2DH12.h>

constexpr uint8_t  PIN_SDA = 21, PIN_SCL = 22;
constexpr uint32_t I2C_FREQ = 400000;
constexpr uint8_t  ADDR_LIS2DH12 = 0x18, ADDR_MCP4531 = 0x2E,
                   ADDR_MCP4018 = 0x2F, ADDR_ADS1115 = 0x48;
constexpr uint8_t  PIN_HV_EN = 18, PIN_LED1 = 16, PIN_LED2 = 17, PIN_DAC_ALC = 25;

constexpr uint32_t PWM_FREQ = 5000;
constexpr uint8_t  PWM_RES_BITS = 8;
constexpr uint32_t SAMPLE_RATE_HZ = 250;
constexpr uint32_t SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE_HZ;
constexpr uint8_t  DEFAULT_ADS_GAIN = 5;
constexpr int16_t  ADS_SINGLE_ENDED_MAX = 32767;

static const adsGain_t ADS_GAIN_TABLE[] = {
    GAIN_TWOTHIRDS, GAIN_ONE, GAIN_TWO, GAIN_FOUR, GAIN_EIGHT, GAIN_SIXTEEN};

Adafruit_ADS1115 ads;
SPARKFUN_LIS2DH12 lis;
bool adsReady = false;
bool lisReady = false;

uint8_t ledLevel1 = 48;
uint8_t ledLevel2 = 48;
uint8_t tiaGain   = 128;
uint8_t boost     = 64;
uint8_t dacOffset = 0;
uint8_t adsGainIdx = DEFAULT_ADS_GAIN;
bool    hvEnabled = false;

char    cmdBuf[64];
uint8_t cmdLen = 0;

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

static void setTiaGain(uint8_t v)   { tiaGain = v;   writeMCP4531(v); }
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

// 3.3 V TIA reference: output falls when photocurrent rises (pulse).
// Remap so the host/GUI sees a conventional PPG (pulse = rising counts).
static int16_t ppgFromAdc(int16_t raw) {
  int32_t r = (int32_t)raw;
  if (r < 0) r = 0;
  if (r > ADS_SINGLE_ENDED_MAX) r = ADS_SINGLE_ENDED_MAX;
  return (int16_t)(ADS_SINGLE_ENDED_MAX - r);
}

static void printStatus() {
  Serial.print("# PONG gain=");   Serial.print(tiaGain);
  Serial.print(" boost=");        Serial.print(boost);
  Serial.print(" dac=");          Serial.print(dacOffset);
  Serial.print(" adsgain=");      Serial.print(adsGainIdx);
  Serial.print(" led1=");         Serial.print(ledLevel1);
  Serial.print(" led2=");         Serial.print(ledLevel2);
  Serial.print(" hv=");           Serial.println(hvEnabled ? 1 : 0);
}

static void handleCommand(char *line) {
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
  else if (!strcmp(line, "HVEN")    && hasValue) setHighVoltage(value != 0);
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
        handleCommand(cmdBuf);
        cmdLen = 0;
      }
    } else if (cmdLen < sizeof(cmdBuf) - 1) {
      cmdBuf[cmdLen++] = c;
    } else {
      cmdLen = 0;
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(200);

  pinMode(PIN_HV_EN, OUTPUT);
  setHighVoltage(false);

  Wire.begin(PIN_SDA, PIN_SCL, I2C_FREQ);
  ledcAttach(PIN_LED1, PWM_FREQ, PWM_RES_BITS);
  ledcAttach(PIN_LED2, PWM_FREQ, PWM_RES_BITS);

  if (ads.begin(ADDR_ADS1115)) {
    adsReady = true;
    ads.setDataRate(RATE_ADS1115_860SPS);
    setAdsGain(DEFAULT_ADS_GAIN);
  } else {
    Serial.println("# ERR: ADS1115 not found");
  }

  setTiaGain(tiaGain);
  setBoost(boost);
  setDacOffset(0);
  setLeds(ledLevel1, ledLevel2);

  if (lis.begin(ADDR_LIS2DH12, Wire)) {
    lisReady = true;
    lis.setScale(2);
    lis.setDataRate(LIS2DH12_ODR_400Hz);
  } else {
    Serial.println("# ERR: LIS2DH12 not found");
  }

  Serial.print("# PPG-SiPM barebones ready, ");
  Serial.print(SAMPLE_RATE_HZ);
  Serial.println(" Hz, ppg polarity corrected @ ADC");
}

void loop() {
  static uint32_t nextSampleUs = 0;
  static int16_t ax = 0, ay = 0, az = 0;

  pollSerial();

  uint32_t now = micros();
  if ((int32_t)(now - nextSampleUs) < 0) return;
  nextSampleUs = now + SAMPLE_PERIOD_US;

  int16_t ppg = ppgFromAdc(readAdc());

  if (lisReady && lis.available()) {
    ax = lis.getX();
    ay = lis.getY();
    az = lis.getZ();
  }

  Serial.printf("%d,%d,%d,%d,%d\n", ppg, ppg, ax, ay, az);
}
