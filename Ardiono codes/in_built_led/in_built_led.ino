void setup() 
{
  // put your setup code here, to run once:
  pinMode(LED_BUILTIN, OUTPUT);
}
int i=0;
void loop() 
{
  // put your main code here, to run repeatedly:


  if(i<=100)
  {
    digitalWrite(LED_BUILTIN,HIGH);
    delay(50);
    digitalWrite(LED_BUILTIN,LOW);
    delay(50);
  }

  if(i>100 && i<=120)
  {
    digitalWrite(LED_BUILTIN,HIGH);
    delay(100);
    digitalWrite(LED_BUILTIN,LOW);
    delay(100);
  }

  if(i>120 && i<=300)
  {
    digitalWrite(LED_BUILTIN,HIGH);
    delay(25);
    digitalWrite(LED_BUILTIN,LOW);
    delay(25);
  }

  i=i+1;
  if(i==300)
  {
    delay(300);
    i=0;
  }
}
