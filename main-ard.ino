#include <Arduino.h>
#include <Wire.h>
#include "Adafruit_SHT31.h"
#include <SPI.h>

Adafruit_SHT31 sht31 = Adafruit_SHT31(); // Inisialisasi sensor SHT31

#define MQ7_PIN A0
#define MQ135_PIN A1
#define MQ136_PIN A2


void disableGasSensors() {
  // Matikan daya ke sensor gas
  digitalWrite(MQ7_PIN, LOW);
  digitalWrite(MQ135_PIN, LOW);
  digitalWrite(MQ136_PIN, LOW);
}

void enableGasSensors() {
  // Nyalakan daya ke sensor gas
  digitalWrite(MQ7_PIN, HIGH);
  digitalWrite(MQ135_PIN, HIGH);
  digitalWrite(MQ136_PIN, HIGH);
}

float convertToPpm(int sensorValue) {
  float ppm = map(sensorValue, 0, 1023, 0, 1000);  // Kalkulasikan ppm sesuai dengan karakteristik sensor
  return ppm;
}

void setup() {
  Serial.begin(9600);
  sht31.begin(0x44);  // Inisialisasi sensor SHT31
  
  pinMode(MQ7_PIN, INPUT);
  pinMode(MQ135_PIN, INPUT);
  pinMode(MQ136_PIN, INPUT);
  
}

void loop() {
  float temperature = sht31.readTemperature();
  float humidity = sht31.readHumidity();

  // Cek kondisi untuk mengaktifkan sensor gas
  if (temperature >= 90 && humidity >= 45 && humidity <= 64) {
    enableGasSensors();
  } else {
    disableGasSensors();
  }

  // Baca data dari sensor gas
  int mq7Value = analogRead(MQ7_PIN);
  int mq135Value = analogRead(MQ135_PIN);
  int mq136Value = analogRead(MQ136_PIN);

  // Konversi data ke ppm
  float mq7Ppm = convertToPpm(mq7Value);
  float mq135Ppm = convertToPpm(mq135Value);
  float mq136Ppm = convertToPpm(mq136Value);

  // Tampilkan data di Serial Monitor dalam format JSON
Serial.print("{");
Serial.print("\"suhu\":");
Serial.print(temperature);
Serial.print(",\"kelembaban\":");
Serial.print(humidity);
Serial.print(",\"mq7Ppm\":");
Serial.print(mq7Ppm);
Serial.print(",\"mq135Ppm\":");
Serial.print(mq135Ppm);
Serial.print(",\"mq136Ppm\":");
Serial.print(mq136Ppm);
Serial.println("}");

  delay(1000);
}
