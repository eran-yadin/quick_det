Here is your **ZeroMQ Quick Start Guide**. This works for any combination (Windows talking to Linux, Linux to Windows, etc.).

### The Architecture

* **Server (The Source):** Binds to a port (e.g., 5555) and broadcasts data.
* **Client (The Receiver):** Connects to the Server's IP and listens.

---

### Phase 1: Preparation (Both Computers)

Run this in your terminal on **both** machines to install the library:

```bash
pip install pyzmq

```

---

### Phase 2: The Server (Computer A)

*This is the machine running the Camera/Detection.*

#### 1. Get your Local IP

You need to know this address to give to the Client later.

* **Windows:** Open Command Prompt → Type `ipconfig` → Look for **IPv4 Address**.
* **Linux:** Open Terminal → Type `hostname -I` or `ip a`.
* *Example Result:* `192.168.1.15`

#### 2. Open the Firewall (Crucial Step)

Your OS will block the connection by default. You must open Port 5555.

* **Windows (PowerShell as Admin):**
```powershell
New-NetFirewallRule -DisplayName "Allow ZMQ" -Direction Inbound -LocalPort 5555 -Protocol TCP -Action Allow

```


*(Or search "Windows Defender Firewall with Advanced Security" → Inbound Rules → New Rule → Port → 5555 → Allow)*
* **Linux (Ubuntu/Debian):**
```bash
sudo ufw allow 5555/tcp

```



#### 3. Run the Server Code

Save as `server_camera.py`:

```python
import zmq
import time

context = zmq.Context()
socket = context.socket(zmq.PUB) # PUB = Publisher (Broadcaster)
socket.bind("tcp://*:5555")      # Listen on ALL interfaces

print("Server started. Broadcasting on Port 5555...")

while True:
    # In real life, this is your camera detection logic
    topic = "alert"
    message = "Person Detected!"
    
    # Send data (Topic + Message)
    socket.send_string(f"{topic} {message}")
    print(f"Sent: {message}")
    time.sleep(1) 

```

---

### Phase 3: The Client (Computer B)

*This is the machine receiving notifications.*

#### 1. Update the IP

Open the code below and replace `192.168.1.X` with the **IP you found in Phase 2**.

#### 2. Run the Client Code

Save as `client_notifier.py`:

```python
import zmq

context = zmq.Context()
socket = context.socket(zmq.SUB) # SUB = Subscriber (Listener)

# REPLACE THIS IP with Computer A's IP address!
socket.connect("tcp://192.168.1.15:5555") 

# Subscribe to the "alert" topic (or "" for everything)
socket.setsockopt_string(zmq.SUBSCRIBE, "alert")

print("Client started. Waiting for server...")

while True:
    # This line blocks until data arrives
    message = socket.recv_string()
    print(f"RECEIVED: {message}")

```

---

### Troubleshooting

* **"Client connects but receives nothing"**:
1. Double-check the IP address in the Client code.
2. Try turning off the Firewall on the **Server** computer temporarily to test.
3. Ensure both computers are on the same Wi-Fi network.


* **"Connection Refused"**: The Server code is not running yet. Start the Server first.

For a visual guide on the trickiest part of this process (Windows settings), you can watch this tutorial:
... [Open Ports in the Windows Firewall](https://www.youtube.com/watch?v=YSkVVn8CZ4E) ...

This video is relevant because opening the correct port in the Windows Firewall is the most common reason why the Client fails to receive messages from the Server.