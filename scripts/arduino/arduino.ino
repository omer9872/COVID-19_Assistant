int r1 = 2;
int g1 = 3;
int b1 = 4;

int r2 = 5;
int g2 = 6;
int b2 = 7;

int r3 = 8;
int g3 = 9;
int b3 = 10;

void setup() {

  pinMode(r1, OUTPUT);
  pinMode(g1, OUTPUT);
  pinMode(b1, OUTPUT);

  pinMode(r2, OUTPUT);
  pinMode(g2, OUTPUT);
  pinMode(b2, OUTPUT);

  pinMode(r3, OUTPUT);
  pinMode(g3, OUTPUT);
  pinMode(b3, OUTPUT);
  
  Serial.begin(9600);

  digitalWrite(r1, HIGH);
  digitalWrite(g1, LOW);
  digitalWrite(b1, LOW);

  digitalWrite(r2, HIGH);
  digitalWrite(g2, LOW);
  digitalWrite(b2, LOW);

  digitalWrite(r3, HIGH);
  digitalWrite(g3, LOW);
  digitalWrite(b3, LOW);

}

void loop() {
  // put your main code here, to run repeatedly:
  char data = (char)Serial.read();

  switch (data) {
    case 'a':
      digitalWrite(r1, HIGH);
      digitalWrite(g1, LOW);
      digitalWrite(b1, HIGH);
      break;
    case 'b':
      digitalWrite(r1, LOW);
      digitalWrite(g1, HIGH);
      digitalWrite(b1, HIGH);
      break;
    case 'c':
      digitalWrite(r1, HIGH);
      digitalWrite(g1, LOW);
      digitalWrite(b1, LOW);
      break;
    case 'd':
      digitalWrite(r2, HIGH);
      digitalWrite(g2, LOW);
      digitalWrite(b2, HIGH);
      break;
    case 'e':
      digitalWrite(r2, LOW);
      digitalWrite(g2, HIGH);
      digitalWrite(b2, HIGH);
      break;
    case 'f':
      digitalWrite(r2, HIGH);
      digitalWrite(g2, LOW);
      digitalWrite(b2, LOW);
      break;
    case 'g':
      digitalWrite(r3, HIGH);
      digitalWrite(g3, LOW);
      digitalWrite(b3, HIGH);
      break;
    case 'h':
      digitalWrite(r3, LOW);
      digitalWrite(g3, HIGH);
      digitalWrite(b3, HIGH);
      break;
    case 'j':
      digitalWrite(r3, HIGH);
      digitalWrite(g3, LOW);
      digitalWrite(b3, LOW);
      break;
  }

  delay(100);
}
