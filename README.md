If you want to implement this, WebRTC via LiveKit (a free, open-source WebRTC platform that has a Python SDK for the Pi and a React SDK for the web app) is usually the easiest production-ready way to get smooth, secure video from an edge device to the web.
Let me know if you'd like to integrate one of these solutions, and I can prepare the web app side to receive the stream!
lightbulb
Thought for 37 seconds
To connect your Raspberry Pi 5 (Edge Device) to this web application, you will need to do two things on your Pi:
Send the AI Procting Alerts directly to your Supabase database.
Run a WebRTC Python script that listens to the web app and streams the camera feed directly to the browser.
Below are the exact steps and code you need to add to your VS Code project on your Raspberry Pi.
Step 1: Install Required Libraries on Raspberry Pi
Open your terminal in VS Code on the Pi and install these dependencies:
code
Bash
pip install aiortc supabase aiohttp websockets opencv-python
Step 2: Update post_alert in proctor_edge.py
In your proctor_edge.py file, you need to modify the post_alert function to include your Supabase API keys so it can insert data directly into your database.
Replace your existing post_alert function (around line 644) with this:
code
Python
import urllib.request
from urllib.request import Request
import json
from typing import Optional, Dict, Any

# Replace these with your actual Supabase Project URL and Anon Key
SUPABASE_URL = "YOUR_SUPABASE_URL"
SUPABASE_KEY = "YOUR_SUPABASE_ANON_KEY"

def post_alert(endpoint: Optional[str], payload: Dict[str, Any]) -> None:
    # We will ignore the endpoint arg and send directly to Supabase
    url = f"{SUPABASE_URL}/rest/v1/alerts"
    
    # Map your Python payload to the Supabase Database columns
    supabase_payload = {
        "session_id": "YOUR_SESSION_ID", # Hardcode or pass down from args
        "type": payload.get("source", "Camera"),
        "message": f"Suspicious activity detected! Score: {payload.get('risk_score', 0)}",
        "severity": "high" if payload.get("risk_score", 0) > 0.8 else ("medium" if payload.get("risk_score", 0) > 0.5 else "low"),
        "timestamp": payload.get("timestamp_iso", "")
    }
    
    body = json.dumps(supabase_payload).encode("utf-8")
    
    req = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Prefer": "return=minimal"
        },
        method="POST",
    )
    
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status not in (200, 201, 204):
                print(f"Failed to post alert: {response.status}")
    except Exception as e:
        print(f"Error posting alert to Supabase: {e}")
Step 3: Create webrtc_stream.py on the Raspberry Pi
Create a new file called webrtc_stream.py in your Raspberry Pi VS Code project. This file will handle the real-time camera connection to the web application.
code
Python
import asyncio
import json
import logging
import cv2
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, VideoStreamTrack
from av import VideoFrame

logging.basicConfig(level=logging.INFO)

# --- CONFIGURATION ---
SUPABASE_URL = "YOUR_SUPABASE_URL" # e.g., https://xyz.supabase.co
SUPABASE_KEY = "YOUR_SUPABASE_ANON_KEY" 
SESSION_ID = "YOUR_SESSION_ID"     # the code you enter on the web app
CAMERA_INDEX = 0                   # 0 for /dev/video0 or Pi Camera

# Convert HTTP URL to WebSocket URL
WS_URL = SUPABASE_URL.replace("https://", "wss://").replace("http://", "ws://")
WS_ENDPOINT = f"{WS_URL}/realtime/v1/websocket?apikey={SUPABASE_KEY}&vsn=1.0.0"

class PiCameraTrack(VideoStreamTrack):
    """
    A video track that returns frames from the Raspberry Pi Camera.
    """
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        # For Raspberry Pi 5, you might want to lower resolution for smoother WebRTC stream
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read from camera. Retrying...")
            await asyncio.sleep(0.1)
            return await self.recv()
            
        # Optional: Add any OpenCV ML drawing here before sending the frame
        
        # Convert BGR to RGB for WebRTC
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        video_frame = VideoFrame.from_ndarray(rgb_frame, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

async def run_webrtc_agent():
    pc = RTCPeerConnection()
    camera_track = PiCameraTrack()
    
    # Supabase Realtime channel setup
    channel_name = f"webrtc_{SESSION_ID}"
    
    async with websockets.connect(WS_ENDPOINT) as websocket:
        print("Connected to Supabase Signaling Server")
        
        # Join the channel
        join_msg = {
            "topic": channel_name,
            "event": "phx_join",
            "payload": {},
            "ref": "1"
        }
        await websocket.send(json.dumps(join_msg))

        async for message in websocket:
            msg = json.loads(message)
            event = msg.get("event")
            payload = msg.get("payload", {})
            
            # --- The web app is asking if we exist ---
            if event == "broadcast" and payload.get("event") == "ping-device":
                print("Received ping from web app. Replying we are ready!")
                reply = {
                    "topic": channel_name,    
                    "event": "broadcast",
                    "payload": {
                        "type": "broadcast",
                        "event": "device-ready",
                        "payload": {}
                    },
                    "ref": "2"
                }
                await websocket.send(json.dumps(reply))

            # --- The web app is ready to view the stream ---
            elif event == "broadcast" and payload.get("event") == "viewer-ready":
                print("Web app is ready. Generating WebRTC Offer...")
                pc.addTrack(camera_track)
                
                offer = await pc.createOffer()
                await pc.setLocalDescription(offer)
                
                offer_msg = {
                    "topic": channel_name,    
                    "event": "broadcast",
                    "payload": {
                        "type": "broadcast",
                        "event": "offer",
                        "payload": {
                            "offer": {
                                "sdp": pc.localDescription.sdp,
                                "type": pc.localDescription.type
                            }
                        }
                    },
                    "ref": "3"
                }
                await websocket.send(json.dumps(offer_msg))

            # --- Receiving Answer from Web App ---
            elif event == "broadcast" and payload.get("event") == "answer":
                print("Received Answer from web app. Setting Remote Description.")
                answer_payload = payload.get("payload", {}).get("answer", {})
                answer = RTCSessionDescription(sdp=answer_payload["sdp"], type=answer_payload["type"])
                await pc.setRemoteDescription(answer)

            # --- Receiving ICE Candidates from Web App ---
            elif event == "broadcast" and payload.get("event") == "candidate":
                candidate_data = payload.get("payload", {}).get("candidate", {})
                if candidate_data:
                    parts = candidate_data["candidate"].split()
                    try:
                        candidate = RTCIceCandidate(
                            component=int(parts[1]),
                            foundation=parts[0],
                            ip=parts[4],
                            port=int(parts[5]),
                            priority=int(parts[3]),
                            protocol=parts[2],
                            type=parts[7],
                            sdpMid=candidate_data.get("sdpMid"),
                            sdpMLineIndex=candidate_data.get("sdpMLineIndex")
                        )
                        await pc.addIceCandidate(candidate)
                    except Exception as e:
                        print(f"Failed parsing candidate: {e}")

        # Block forever
        await asyncio.Event().wait()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(run_webrtc_agent())
    except KeyboardInterrupt:
        pass
Step 4: Run It!
Open your src/supabaseClient.ts in the Web Application Code Editor to copy your Supabase URL and Supabase Anon Key and paste them into both proctor_edge.py and webrtc_stream.py.
Start the web application and click "Start Session".
The web app will give you a Session ID code. Paste that into SESSION_ID in webrtc_stream.py (and use it for proctor_edge.py as well!).
Run your python scripts on the Pi! proctor_edge.py will send the motion and ML alerts directly to your database, and python webrtc_stream.py will stream the camera straight to your dashboard over WebRTC!
(Optional Tip: You can import and run the WebRTC loop inside a Thread in your main run_proctor_menu.py or run webrtc_stream.py directly alongside it).
