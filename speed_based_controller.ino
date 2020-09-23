#include <LiquidCrystal.h>
#include <Wire.h>
#include <Dwenguino.h>
//#include <Sabertooth.h>


const unsigned int BAUD_RATE = 38400;
const unsigned int BAUDING_CHARACTER = 170; // (0B10101010)
const unsigned int SABERTOOTH_ADDRESS = 128;
const unsigned int FORWARD_COMMAND    = 0;
const unsigned int BACKWARD_COMMAND   = 1;
const unsigned int MAXVOLT_COMMAND    = 3;
const unsigned int MAXVOLT_VALUE      = 127;
const int MAX_VELOCITY = 127;
const int MIN_VELOCITY = -127;

int CURRENT_VELOCITY = 0;
int TARGET_VELOCITY = 0;
int DELTA_TIME  = 2; // amount of time in milliseconds between hard-set motor commands                           2
int DELTA_VELOCITY = 10; // maximum change in velocity per DELTA_TIME on hard-set motor commands                  3
long PREV_VEL_UPDATE = 0;
long PREV_SPD_REPORT = 0;
int DELTA_REPORT_TIME = 10;

bool PRINT_STRING_MODE = true;
int  PRINT_INT_VALUE = -99;
String PRINT_STRING_VALUE = "";



void print(String msg) {
//  return;
  if( PRINT_STRING_MODE ) {
    if( msg.compareTo(PRINT_STRING_VALUE) != 0) {
      dwenguinoLCD.clear();
      dwenguinoLCD.print(msg);
      PRINT_STRING_VALUE = msg;
    }
  } else {
    dwenguinoLCD.clear();
    dwenguinoLCD.print(msg);
    PRINT_STRING_MODE = true;
    PRINT_STRING_VALUE = msg;
  }
}

void print(long msg) {
//  return;
  if( !PRINT_STRING_MODE ) {
    if( msg != PRINT_INT_VALUE ) {
      dwenguinoLCD.clear();
      dwenguinoLCD.print(msg);
      PRINT_INT_VALUE = msg;
    }
  } else {
    dwenguinoLCD.clear();
    dwenguinoLCD.print(msg);
    PRINT_STRING_MODE = false;
    PRINT_INT_VALUE = msg;
  }
}

void maxVoltage(){
  Serial1.write(SABERTOOTH_ADDRESS);
  Serial1.write(MAXVOLT_COMMAND);
  Serial1.write(MAXVOLT_VALUE);
  Serial1.write((SABERTOOTH_ADDRESS + MAXVOLT_COMMAND + MAXVOLT_VALUE) & 0b01111111); //checksum
}

void hardSetSpeed(int velocity) {
  if (velocity >= 0) { // move forwards
    Serial1.write(SABERTOOTH_ADDRESS);
    Serial1.write(FORWARD_COMMAND);
    Serial1.write(velocity);
    Serial1.write((SABERTOOTH_ADDRESS + FORWARD_COMMAND + velocity) & 0b01111111); // checksum
  } else { // move backwards
    Serial1.write(SABERTOOTH_ADDRESS);
    Serial1.write(BACKWARD_COMMAND); // <---------------------------,
    Serial1.write(-velocity); // inverse: <velocity = -42>  ->  <backward 42>
    Serial1.write((SABERTOOTH_ADDRESS + BACKWARD_COMMAND - velocity) & 0b01111111); //checksum
  }
}

// set target speed, clipped at min and max velocity
void softSetSpeed(int velocity) {
  if (MIN_VELOCITY <= velocity && MAX_VELOCITY >= velocity) {
    TARGET_VELOCITY = velocity;
  } else {
    TARGET_VELOCITY = 0;
  }
}

// move current speed closer to target speed (low pass filter)
void updateSpeed(long currentMillis) {
  if (CURRENT_VELOCITY == TARGET_VELOCITY) {return;}
  if (currentMillis - PREV_VEL_UPDATE < DELTA_TIME) {return;}
  
//  print(String(CURRENT_VELOCITY) + " " + String(TARGET_VELOCITY));
  if (CURRENT_VELOCITY < TARGET_VELOCITY) {
    CURRENT_VELOCITY = min(CURRENT_VELOCITY + DELTA_VELOCITY, TARGET_VELOCITY);
  } else {
    CURRENT_VELOCITY = max(CURRENT_VELOCITY - DELTA_VELOCITY, TARGET_VELOCITY);
  }
  hardSetSpeed(CURRENT_VELOCITY); 
  PREV_VEL_UPDATE = currentMillis;
}

void reportSpeed(long currentMillis) {
  if (currentMillis - PREV_SPD_REPORT < DELTA_REPORT_TIME) {return;}
  if (Serial.availableForWrite()) {
    Serial.println(CURRENT_VELOCITY);
  }

  PREV_SPD_REPORT = currentMillis;
}

void waitForButtonPress() {
  while( digitalRead(SW_W) != PRESSED) {
    dwenguinoLCD.clear();
    dwenguinoLCD.print("Press SW_W");
    delay(20);
  }
  print("ok");
}


void initSerial1() {
  Serial1.begin(BAUD_RATE);
  while (!Serial1) {;}
  hardSetSpeed(0); // if motor was still running, stop it immediately
  delay(2000);
  Serial1.write(BAUDING_CHARACTER); //send bauding character (0b10101010)
  delay(100);
  Serial1.flush();
  delay(2000);
//  maxVoltage();
  delay(100);
  Serial1.flush();
}

void initSerial(){
  Serial.begin(9600);
  Serial.setTimeout(10);
}

void setup() {
  initDwenguino();
  initSerial1();
  initSerial();
  waitForButtonPress();
}

long now = millis();
long prev_msg = millis();
void loop() {
  now = millis();
  if(Serial.available()) {
    int cmd = 0;
    while(Serial.available()) {
      cmd = Serial.parseInt();
    }
    softSetSpeed(cmd);
    prev_msg = millis();
  } else if (now - 50 > prev_msg) {
    softSetSpeed(0);
  }
  updateSpeed(now);
  reportSpeed(now);
}
