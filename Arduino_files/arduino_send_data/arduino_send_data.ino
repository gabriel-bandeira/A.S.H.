/*
 * Arduino send data through Blutooth
 * 
 * The value of sensors are read, concatened on 
 * String and send via serial port.
 */
 
#include "DHT.h"

#define DHTPIN	3
#define DHTTYPE	DHT22
#define PIRPIN	9
#define COPIN	A6

DHT dht(DHTPIN, DHTTYPE);
float humidaty, temperature;
boolean pir = 0;
int co, mic;

String msg = "";
char nome[40];

void setup() {
  Serial.begin(9600);
  dht.begin();
}

void loop() {
  humidaty = dht.readHumidity();
  temperature = dht.readTemperature();
  pir = digitalRead(PIRPIN);
  co = analogRead(COPIN);
  mic = analogRead(A0); 
  msg = "#;" +String(humidaty) + ";" + String(temperature) +";"+ String(mic) +";"+ String(pir)+ ";" + String(co) + ";#" +"\n";
  Serial.print(msg);
  delay(2000);
}

