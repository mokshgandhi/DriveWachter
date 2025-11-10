const int trigPin = 9;
const int echoPin = 10;
float duration, distance;
const float x = 30.0; 
float potholeDepth;

void setup() {
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  Serial.begin(9600);
}

void loop() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  duration = pulseIn(echoPin, HIGH);
  distance = (duration * 0.0343) / 2;
  potholeDepth = distance - x;
  if (potholeDepth > 5) {
    Serial.print("Pothole Depth: ");
    Serial.print(potholeDepth);
    Serial.println(" cm");
  } else {
    Serial.println("No pothole detected (flat or raised surface).");
  }
  delay(500);
}
