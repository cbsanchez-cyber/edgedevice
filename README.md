# supabase_webrtc.py
import asyncio
import json
import uuid
import fractions
import time
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

class QueueVideoStreamTrack(VideoStreamTrack):
    """
    A video stream track that reads frames from a thread-safe Queue.
    """
    kind = "video"

    def __init__(self, frame_queue):
        super().__init__()
        self.frame_queue = frame_queue

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        
        # Get frame from synchronous proctor_edge.py loop
        # We use a small sleep to not block the asyncio loop while waiting
        frame = None
        while frame is None:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
            else:
                await asyncio.sleep(0.01)

        # Convert OpenCV BGR image to PyAV VideoFrame
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame


async def run_webrtc_signaling(session_id, supabase_url, supabase_anon_key, frame_queue):
    # Extract WebSocket URL from your Supabase URL
    ws_url = supabase_url.replace("http", "ws") + f"/realtime/v1/websocket?apikey={supabase_anon_key}&vsn=1.0.0"
    topic = f"webrtc_{session_id}"
    join_ref = 1
    
    pc = RTCPeerConnection()
    pc.addTrack(QueueVideoStreamTrack(frame_queue))

    async with websockets.connect(ws_url) as websocket:
        # 1. Join the Realtime Channel
        join_msg = {
            "topic": f"realtime:{topic}",
            "event": "phx_join",
            "payload": {"config": {"broadcast": {"ack": False, "self": False}}},
            "ref": str(join_ref)
        }
        await websocket.send(json.dumps(join_msg))

        # Helper method for sending broadcasts
        async def send_broadcast(event, payload):
            nonlocal join_ref
            join_ref += 1
            msg = {
                "topic": f"realtime:{topic}",
                "event": "broadcast",
                "payload": {
                    "type": "broadcast",
                    "event": event,
                    "payload": payload
                },
                "ref": str(join_ref)
            }
            await websocket.send(json.dumps(msg))

        # Handle Responses
        async for message in websocket:
            data = json.loads(message)
            if data.get("event") == "phx_reply" and data.get("ref") == "1":
                print(f"[WebRTC] Subscribed to {topic}")
                # We tell the dashboard we are online
                await send_broadcast("device-ready", {})

            elif data.get("event") == "broadcast":
                # Extract inner payload from Supabase broadcast
                inner_event = data.get("payload", {}).get("event")
                inner_payload = data.get("payload", {}).get("payload", {})

                if inner_event == "ping-device":
                    # Let the Web App know we're here
                    print(f"[WebRTC] Received Ping, responding with device-ready")
                    await send_broadcast("device-ready", {})
                
                elif inner_event == "viewer-ready":
                    # Web application clicked "Begin Proctoring", create the WebRTC offer
                    print(f"[WebRTC] Viewer ready, creating WebRTC Offer...")
                    offer = await pc.createOffer()
                    await pc.setLocalDescription(offer)
                    await send_broadcast("offer", {
                        "offer": {"type": pc.localDescription.type, "sdp": pc.localDescription.sdp}
                    })

                elif inner_event == "answer":
                    # Handle WebRTC Answer from the frontend
                    print(f"[WebRTC] Received Answer, establishing P2P connection...")
                    answer = RTCSessionDescription(sdp=inner_payload["answer"]["sdp"], type=inner_payload["answer"]["type"])
                    await pc.setRemoteDescription(answer)


def start_webrtc_thread(session_id, supabase_url, supabase_anon_key, frame_queue):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_webrtc_signaling(session_id, supabase_url, supabase_anon_key, frame_queue))
    except Exception as e:
        print(f"[WebRTC] Connection Closed: {e}")
