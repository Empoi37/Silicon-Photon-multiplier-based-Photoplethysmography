/*
 * ESP32 PPG-SiPM sensor firmware
 *
 * Two green LEDs illuminate the tissue continuously and an ADS1115 samples the
 * transimpedance-amplifier (TIA) output. PPG + 3-axis accelerometer samples are
 * streamed to the host as CSV; the host tunes the analog front-end over serial.
 *
 * I2C bus (SDA=21, SCL=22):
 *   0x18  LIS2DH12  3-axis accelerometer     (SparkFun_LIS2DH12)
 *   0x2E  MCP4531   TIA gain digipot         (raw I2C)
 *   0x2F  MCP4018   SiPM overvoltage digipot (raw I2C)
 *   0x48  ADS1115   16-bit ADC, TIA output   (Adafruit_ADS1X15)
 *
 * GPIO:
 *   18  HV_EN    high-voltage enable (LOW = off / safe)
 *   16  LED1     green LED #1, PWM via MOSFET
 *   17  LED2     green LED #2, PWM via MOSFET
 *   25  DAC_ALC  8-bit DAC, DC-offset injection at the amplifier input
 *
 * Serial protocol @115200:
 *   ESP32 -> host : "led1,led2,ax,ay,az\n"   (both PPG columns share one channel)
 *   host  -> ESP32: "CMD:VALUE\n"
 *     GAIN:0-255  BOOST:0-127  DAC:0-255  ADSGAIN:0-5
 *     LED1:0-255  LED2:0-255   HVEN:0|1   DCSERVO:0|1   PING
 */

#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <SparkFun_LIS2DH12.h>

// ---- Hardware map ----
constexpr uint8_t  PIN_SDA = 21, PIN_SCL = 22;
constexpr uint32_t I2C_FREQ = 400000;
constexpr uint8_t  ADDR_LIS2DH12 = 0x18, ADDR_MCP4531 = 0x2E,
                   ADDR_MCP4018 = 0x2F, ADDR_ADS1115 = 0x48;
constexpr uint8_t  PIN_HV_EN = 18, PIN_LED1 = 16, PIN_LED2 = 17, PIN_DAC_ALC = 25;

// ---- Acquisition ----
constexpr uint32_t PWM_FREQ = 5000;
constexpr uint8_t  PWM_RES_BITS = 8;
constexpr uint32_t SAMPLE_RATE_HZ = 250;
constexpr uint32_t SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE_HZ;
constexpr uint8_t  DEFAULT_ADS_GAIN = 5;   // index 5 = +/-256 mV, ~7.8 uV/LSB

// ADS1115 full-scale per gain index; higher index = finer resolution.
static const adsGain_t ADS_GAIN_TABLE[] = {
    GAIN_TWOTHIRDS, GAIN_ONE, GAIN_TWO, GAIN_FOUR, GAIN_EIGHT, GAIN_SIXTEEN};

// ---- Software DC servo (optional) ----
// Nudges the DAC to keep the DC level near mid-scale. Note: on this board the
// DAC only shifts the reading by ~15 counts full-swing, so it cannot fully
// cancel the DC. Kept as an option; DC is normally removed on the host side.
constexpr float    DC_SERVO_TARGET   = 16000.0f;
constexpr float    DC_SERVO_ALPHA    = 0.004f;    // ~0.15 Hz low-pass at 250 Hz
constexpr float    DC_SERVO_KP       = 0.0008f;   // error (counts) -> DAC steps
constexpr int      DC_SERVO_MAX_STEP = 3;
constexpr uint32_t DC_SERVO_PERIOD_US = 40000;    // update DAC at ~25 Hz

// ---- Devices ----
Adafruit_ADS1115 ads;
SPARKFUN_LIS2DH12 lis;
bool adsReady = false;
bool lisReady = false;

// ---- Front-end state (mirrors host sliders) ----
uint8_t ledLevel1  = 48;
uint8_t ledLevel2  = 48;
uint8_t tiaGain    = 128;
uint8_t boost      = 64;
uint8_t dacOffset  = 128;
uint8_t adsGainIdx = DEFAULT_ADS_GAIN;
bool    hvEnabled  = false;

// ---- DC servo state ----
bool     dcServoEnabled = false;
float    dcEstimate     = DC_SERVO_TARGET;
int8_t   dcServoSign    = +1;              // DAC->ADC polarity, set on enable
uint32_t dcLastUpdateUs = 0;

// ---- Serial command buffer ----
char    cmdBuf[64];
uint8_t cmdLen = 0;

// ---------------------------------------------------------------------------
// Digipots (raw I2C)
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

// ---------------------------------------------------------------------------
// Actuators
// ---------------------------------------------------------------------------
static void setTiaGain(uint8_t v)   { tiaGain = v;      writeMCP4531(v); }
static void setBoost(uint8_t v)     { boost = v & 0x7F; writeMCP4018(boost); }
static void setDacOffset(uint8_t v) { dacOffset = v;    dacWrite(PIN_DAC_ALC, v); }
static void setHighVoltage(bool en) { hvEnabled = en;   digitalWrite(PIN_HV_EN, en ? HIGH : LOW); }

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
  // Re-arm continuous conversion so the new gain takes effect.
  ads.startADCReading(ADS1X15_REG_CONFIG_MUX_SINGLE_0, /*continuous=*/true);
}

static int16_t readAdc() {
  return adsReady ? ads.getLastConversionResults() : 0;
}

// ---------------------------------------------------------------------------
// DC servo
// ---------------------------------------------------------------------------
// Learn whether raising the DAC raises or lowers the ADC reading.
static void calibrateDcServo() {
  if (!adsReady) return;
  setDacOffset(90);  delay(120);
  long lo = 0; for (int i = 0; i < 8; ++i) { lo += readAdc(); delay(3); }
  setDacOffset(170); delay(120);
  long hi = 0; for (int i = 0; i < 8; ++i) { hi += readAdc(); delay(3); }
  dcServoSign = (hi >= lo) ? +1 : -1;
  setDacOffset(128);
  dcEstimate = DC_SERVO_TARGET;
}

// Update the slow DC estimate every sample; move the DAC when the servo is on.
static void updateDcServo(int16_t adc, uint32_t now) {
  dcEstimate += DC_SERVO_ALPHA * ((float)adc - dcEstimate);

  if (!dcServoEnabled) return;
  if ((uint32_t)(now - dcLastUpdateUs) < DC_SERVO_PERIOD_US) return;
  dcLastUpdateUs = now;

  int step = (int)(DC_SERVO_KP * (DC_SERVO_TARGET - dcEstimate));
  step = constrain(step, -DC_SERVO_MAX_STEP, DC_SERVO_MAX_STEP);
  int next = constrain((int)dacOffset + dcServoSign * step, 0, 255);
  if (next != (int)dacOffset) setDacOffset((uint8_t)next);
}

// ---------------------------------------------------------------------------
// Serial commands
// ---------------------------------------------------------------------------
static void printStatus() {
  Serial.print("# PONG gain=");   Serial.print(tiaGain);
  Serial.print(" boost=");        Serial.print(boost);
  Serial.print(" dac=");          Serial.print(dacOffset);
  Serial.print(" adsgain=");      Serial.print(adsGainIdx);
  Serial.print(" led1=");         Serial.print(ledLevel1);
  Serial.print(" led2=");         Serial.print(ledLevel2);
  Serial.print(" dcservo=");      Serial.print(dcServoEnabled ? 1 : 0);
  Serial.print(" dcest=");        Serial.print((int)dcEstimate);
  Serial.print(" hv=");           Serial.println(hvEnabled ? 1 : 0);
}

// Parse and apply one "CMD:VALUE" line.
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
  else if (!strcmp(line, "DAC")     && hasValue) setDacOffset(constrain(value, 0, 255));
  else if (!strcmp(line, "ADSGAIN") && hasValue) setAdsGain(constrain(value, 0, 5));
  else if (!strcmp(line, "LED1")    && hasValue) setLeds(constrain(value, 0, 255), ledLevel2);
  else if (!strcmp(line, "LED2")    && hasValue) setLeds(ledLevel1, constrain(value, 0, 255));
  else if (!strcmp(line, "HVEN")    && hasValue) setHighVoltage(value != 0);
  else if (!strcmp(line, "DCSERVO") && hasValue) {
    dcServoEnabled = (value != 0);
    if (dcServoEnabled) calibrateDcServo();
  }
  else if (!strcmp(line, "PING")) printStatus();
  else { Serial.print("# ERR unknown cmd: "); Serial.println(line); }
}

// Accumulate serial bytes into cmdBuf and dispatch on newline.
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
      cmdLen = 0;  // overflow: drop the line
    }
  }
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------
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
    ads.setDataRate(RATE_ADS1115_860SPS);   // max rate; drives our 250 Hz loop
    setAdsGain(DEFAULT_ADS_GAIN);            // also starts continuous conversion
  } else {
    Serial.println("# ERR: ADS1115 not found (0x48)");
  }

  setTiaGain(tiaGain);
  setBoost(boost);
  setDacOffset(dacOffset);
  setLeds(ledLevel1, ledLevel2);             // LEDs stay on continuously

  if (lis.begin(ADDR_LIS2DH12, Wire)) {
    lisReady = true;
    lis.setScale(2);
    // setDataRate expects an ODR enum, not Hz. Max ODR prevents getX() from
    // blocking (data-ready wait), which would otherwise throttle the loop.
    lis.setDataRate(LIS2DH12_ODR_400Hz);
  } else {
    Serial.println("# ERR: LIS2DH12 not found (0x18)");
  }

  Serial.print("# ESP32 PPG-SiPM ready. Continuous LEDs, ");
  Serial.print(SAMPLE_RATE_HZ);
  Serial.println(" Hz, HV off.");
}

// ---------------------------------------------------------------------------
// Main loop: stream one CSV sample every SAMPLE_PERIOD_US
// ---------------------------------------------------------------------------
void loop() {
  static uint32_t nextSampleUs = 0;
  static int16_t ax = 0, ay = 0, az = 0;

  pollSerial();

  uint32_t now = micros();
  if ((int32_t)(now - nextSampleUs) < 0) return;
  nextSampleUs = now + SAMPLE_PERIOD_US;

  // PPG: latest continuous conversion (non-blocking). LEDs are always on, so
  // every conversion is valid and there is no LED-switching artifact.
  int16_t ppg = readAdc();
  updateDcServo(ppg, now);

  // Accelerometer: read only when a new sample is ready so getX() never blocks.
  if (lisReady && lis.available()) {
    ax = lis.getX();
    ay = lis.getY();
    az = lis.getZ();
  }

  // Both PPG columns carry the same combined-LED channel (host expects 5 cols).
  char out[48];
  snprintf(out, sizeof(out), "%d,%d,%d,%d,%d", ppg, ppg, ax, ay, az);
  Serial.println(out);
}
