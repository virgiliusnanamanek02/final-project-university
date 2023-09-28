#include <Arduino.h>
#include <Wire.h>
#include "Adafruit_SHT31.h"
#include <SPI.h>

Adafruit_SHT31 sht31 = Adafruit_SHT31(); // Inisialisasi sensor SHT31

#define MQ7_PIN A0
#define MQ135_PIN A1
#define MQ136_PIN A2

bool gasSensorsActive = false; // Status sensor gas aktif atau tidak
unsigned long lastActivationTime = 0; // Deklarasikan variabel lastActivationTime

void disableGasSensors() {
  // Matikan daya ke sensor gas
  digitalWrite(MQ7_PIN, LOW);
  digitalWrite(MQ135_PIN, LOW);
  digitalWrite(MQ136_PIN, LOW);
  gasSensorsActive = false;
}

void enableGasSensors() {
  // Nyalakan daya ke sensor gas
  digitalWrite(MQ7_PIN, HIGH);
  digitalWrite(MQ135_PIN, HIGH);
  digitalWrite(MQ136_PIN, HIGH);
  gasSensorsActive = true;
}

float convertToPpm(int sensorValue) {
  // Konversi dari nilai sensor ke ppm sesuai dengan karakteristik sensor gas yang digunakan
  // Anda perlu mengacu pada datasheet masing-masing sensor untuk perhitungan ppm yang sesuai
  // Contoh: Kalkulasikan ppm berdasarkan data sheet sensor
  float ppm = map(sensorValue, 0, 1023, 0, 1000);  // Kalkulasikan ppm sesuai dengan karakteristik sensor
  return ppm;
}

void setup() {
  Serial.begin(9600);
  sht31.begin(0x44);  // Inisialisasi sensor SHT31
  
  pinMode(MQ7_PIN, OUTPUT); // Ubah pin MQ7 ke OUTPUT
  pinMode(MQ135_PIN, INPUT);
  pinMode(MQ136_PIN, OUTPUT); // Ubah pin MQ136 ke OUTPUT
}

void loop() {
  float temperature = sht31.readTemperature();
  float humidity = sht31.readHumidity();

  // Cek kondisi untuk mengaktifkan sensor gas
  if (temperature >= 20 && humidity >= 55 && humidity <= 75) {
    // Aktifkan sensor gas hanya jika lebih dari 10 detik telah berlalu sejak terakhir diaktifkan
    if (!gasSensorsActive && millis() - lastActivationTime >= 10000) {
      enableGasSensors();
      lastActivationTime = millis(); // Perbarui waktu terakhir sensor diaktifkan
    }
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

  // Tampilkan data di Serial Monitor
  Serial.print("{");
  Serial.print("\"suhu\":");
  Serial.print(temperature);
  Serial.print(",\"kelembaban\":");
  Serial.print(humidity);
   
  // Cetak nilai sensor gas jika sensor gas aktif, atau "Belum Aktif" jika tidak aktif
  Serial.print(",\"mq7Ppm\":");
  if (gasSensorsActive) {
    Serial.print(mq7Ppm);
  } else {
    Serial.print("Belum Aktif");
  }
  
  Serial.print(",\"mq135Ppm\":");
  if (gasSensorsActive) {
    Serial.print(mq135Ppm);
  } else {
    Serial.print("Belum Aktif");
  }
  
  Serial.print(",\"mq136Ppm\":");
  if (gasSensorsActive) {
    Serial.print(mq136Ppm);
  } else {
    Serial.print("Belum Aktif");
  }
  
  Serial.println("}");

  delay(1000);
}

