**Warning:** You cannot connect them directly with a wire!

* **Raspberry Pi** uses **3.3V** logic.
* **Zebra FX7500** GPIO inputs are often pulled up to **5V** or **24V** (Industrial standard).
* **Result of direct connection:** You will fry your Raspberry Pi immediately.

You need a **Relay Module** (cost: ~$2) to act as a safe bridge.

### The Hardware Setup (The Safety Bridge)

The Relay isolates the two machines. The Pi triggers the Relay, and the Relay "presses the button" on the Zebra.

#### 1. Wiring the Raspberry Pi to the Relay

Get a standard **5V Relay Module** (compatible with Arduino/Pi).

* **VCC** → Pi 5V Pin
* **GND** → Pi GND
* **IN** → Pi GPIO 17 (or any free pin)

#### 2. Wiring the Relay to the Zebra FX7500

The Zebra FX7500 inputs are "Active Low" (usually). This means they trigger when you connect the Input Pin to the Ground Pin.

* **Relay COM (Common)** → Zebra **GND** (Pin 8 or 5)
* **Relay NO (Normally Open)** → Zebra **GPI 1** (Pin 6)

### The Python Code (Raspberry Pi)

This script detects a person (simulated) and "closes the switch" on the Zebra reader for 1 second.

```python
import RPi.GPIO as GPIO
import time

# Setup
RELAY_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)

# Ensure Relay is OFF initially
# (Most relays are 'Active Low', meaning Low=ON, High=OFF. 
# If yours is different, swap True/False)
GPIO.output(RELAY_PIN, GPIO.HIGH) 

print("System Ready. Waiting for detection...")

try:
    while True:
        # --- YOUR DETECTION LOGIC HERE ---
        person_detected = True # Replace with your CV logic
        
        if person_detected:
            print("Person found! Signaling Zebra...")
            
            # 1. Turn Relay ON (Connects Zebra Pin 6 to GND)
            GPIO.output(RELAY_PIN, GPIO.LOW) 
            
            # 2. Hold signal for a moment (so Zebra sees it)
            time.sleep(0.5) 
            
            # 3. Turn Relay OFF
            GPIO.output(RELAY_PIN, GPIO.HIGH)
            
            # Wait before next scan
            time.sleep(2) 
            
except KeyboardInterrupt:
    GPIO.cleanup()

```

### The Zebra Side (Configuration)

You don't need code on the Zebra, just configuration.

1. Log into the Zebra Web Console (usually `http://fx7500<serial>.local` or its IP).
2. Go to **GPIO Settings**.
3. Look for **GPI 1 (Input 1)**.
4. You can map this input to an action (e.g., "Start Reading when Input is Low" or "Stop Reading when Input is Low").

### Alternative: The "Cable Only" Way (No GPIO Wires)

Since you mentioned both devices have "cable connections" (Ethernet), you can technically skip the GPIO wires and send a command over the network using the **LLRP (Low Level Reader Protocol)**.

* **Pros:** No soldering/relays.
* **Cons:** Much harder to program. You need a library like `sllurp` in Python to send a "Start RO Spec" command.
* **Recommendation:** Stick to the Relay method (GPIO). It is easier to debug: you can hear the relay "click" when it works.

... [How to Control Lights, Buzzers and More with RFID Reader GPIO Settings](https://www.youtube.com/watch?v=2PGMyngFVjo) ...

This video is relevant because it shows the exact interface inside the Zebra software where you configure what happens when the GPIO pins are triggered (like the signal your Pi will send).