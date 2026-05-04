import argparse
import asyncio
import csv
import json
import math
import os
import platform
import queue
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import TextIOWrapper
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Optional: aiortc (WebRTC video streaming) + websockets (Supabase Realtime)
# Install with:  pip install aiortc websockets
# If not installed the script still works – live video streaming is disabled.
# ---------------------------------------------------------------------------
try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
    from aiortc.contrib.media import MediaBlackhole
    import fractions
    import av
    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

try:
    import torch
except Exception:
    torch = None

PICAMERA2_DIST_PATH = "/usr/lib/python3/dist-packages"
if os.path.isdir(PICAMERA2_DIST_PATH) and PICAMERA2_DIST_PATH not in sys.path:
    sys.path.append(PICAMERA2_DIST_PATH)

try:
    import importlib

    _picamera2_mod = importlib.import_module("picamera2")
    Picamera2 = getattr(_picamera2_mod, "Picamera2", None)
except Exception:
    Picamera2 = None


PERSON_CLASS_ID = 0
CONTRABAND_CLASS_IDS = [63, 67, 73]
CONTRABAND_WEIGHTS = {
    "cell phone": 0.40,
    "book": 0.25,
    "laptop": 0.20,
}

HEAD_RISK_LEFT_RIGHT = 0.35
HEAD_RISK_DOWN = 0.25
MOTION_THRESHOLD = 35.0
PROXIMITY_THRESHOLD = 120.0
CONTRABAND_ASSOCIATION_THRESHOLD = 180.0
DEFAULT_ALERT_THRESHOLD = 0.75

RISK_NORMAL_MAX = 0.25
RISK_REVIEW_MAX = 0.75

COLOR_GREEN = (0, 200, 0)
COLOR_ORANGE = (0, 165, 255)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_CYAN = (255, 255, 0)

EVENT_HEADER = [
    "event_ts_iso",
    "event_ts_ms",
    "frame_index",
    "student_id",
    "event_type",
    "risk_score",
    "head_status",
    "alert_threshold",
    "details",
]

REPORT_HEADER = [
    "student_id",
    "samples",
    "avg_risk",
    "max_risk",
    "final_label",
    "image_path",
    "image_captured_at_ms",
]


@dataclass
class EventLogHandle:
    fh: TextIOWrapper
    writer: Any


@dataclass
class RuntimeConfig:
    source: Union[int, str]
    source_raw: str
    output: Optional[str]
    width: int
    height: int
    det_imgsz: int
    pose_imgsz: int
    det_conf: float
    contraband_interval: int
    pose_interval: int
    headless: bool
    fast: bool
    report_csv: str
    event_log_csv: str
    report_image_dir: str
    alert_endpoint: Optional[str]
    alert_threshold: float
    alert_cooldown_sec: float
    raspi_mode: bool
    camera_fps: int
    status_every_frames: int
    frame_fit: str
    # --- GuardEye Supabase integration (NEW) ---
    supabase_url: Optional[str] = None       # e.g. https://xxxx.supabase.co
    supabase_anon_key: Optional[str] = None  # project anon/public key
    session_id: Optional[str] = None         # GuardEye session UUID
    webrtc_enabled: bool = False             # stream annotated video via WebRTC
    device_id: str = "pi-edge-001"           # shown in Hardware Connection dialog


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def parse_source(source: str) -> Union[int, str]:
    if source.strip().lstrip("+-").isdigit():
        return int(source.strip())
    return source


def is_raspi_camera_source(source: Union[int, str]) -> bool:
    if not isinstance(source, str):
        return False
    normalized = source.strip().lower()
    return normalized in {"pi", "raspi", "picam", "libcamera"}


class Picamera2CaptureWrapper:
    def __init__(self, width: int, height: int, fps: int) -> None:
        if Picamera2 is None:
            raise RuntimeError("picamera2 is not available in this Python environment")

        self._width = width
        self._height = height
        self._fps = fps
        self._opened = False
        self._cam = Picamera2()

        config = self._cam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": float(fps)},
        )
        self._cam.configure(config)
        self._cam.start()
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._opened:
            return False, None
        try:
            rgb = self._cam.capture_array("main")
        except Exception:
            return False, None
        if rgb is None:
            return False, None
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return True, bgr

    def release(self) -> None:
        if not self._opened:
            return
        try:
            self._cam.stop()
        except Exception:
            pass
        self._opened = False

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self._fps)
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        return 0.0


def open_picamera2_capture(cfg: RuntimeConfig) -> Optional[Picamera2CaptureWrapper]:
    if Picamera2 is None:
        return None
    try:
        cap = Picamera2CaptureWrapper(cfg.width, cfg.height, cfg.camera_fps)
        if cap.isOpened():
            print("Using Picamera2 capture backend")
            return cap
    except Exception as exc:
        print(f"Picamera2 init failed: {exc}")
    return None


def detect_raspberry_pi() -> bool:
    if platform.system().lower() != "linux":
        return False

    model_path = "/proc/device-tree/model"
    cpuinfo_path = "/proc/cpuinfo"

    try:
        if os.path.exists(model_path):
            with open(model_path, "r", encoding="utf-8", errors="ignore") as fh:
                model = fh.read().strip().lower()
                if "raspberry pi" in model:
                    return True
    except OSError:
        pass

    try:
        if os.path.exists(cpuinfo_path):
            with open(cpuinfo_path, "r", encoding="utf-8", errors="ignore") as fh:
                cpuinfo = fh.read().lower()
                if "raspberry pi" in cpuinfo or "bcm27" in cpuinfo:
                    return True
    except OSError:
        pass

    return False


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="Edge AI Proctoring with YOLOv8 + DeepSORT")
    parser.add_argument("--source", default="0")
    parser.add_argument("--output", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--det-imgsz", type=int, default=640)
    parser.add_argument("--pose-imgsz", type=int, default=640)
    parser.add_argument("--det-conf", type=float, default=0.35)
    parser.add_argument("--contraband-interval", type=int, default=1)
    parser.add_argument("--pose-interval", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--report-csv", default="student_report.csv")
    parser.add_argument("--event-log-csv", default="event_log.csv")
    parser.add_argument("--report-image-dir", default="student_images")
    parser.add_argument("--alert-endpoint", default=None)
    parser.add_argument("--alert-threshold", type=float, default=DEFAULT_ALERT_THRESHOLD)
    parser.add_argument("--alert-cooldown-sec", type=float, default=5.0)
    parser.add_argument("--raspi", action="store_true")
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--show-window", action="store_true")
    parser.add_argument("--status-every-frames", type=int, default=30)
    parser.add_argument("--frame-fit", choices=["stretch", "contain", "cover"], default="stretch")
    # --- GuardEye Supabase integration (NEW) ---
    parser.add_argument(
        "--supabase-url", default=os.environ.get("SUPABASE_URL"),
        help="Supabase project URL, e.g. https://xxxx.supabase.co  (or set SUPABASE_URL env var)",
    )
    parser.add_argument(
        "--supabase-anon-key", default=os.environ.get("SUPABASE_ANON_KEY"),
        help="Supabase anon/public API key  (or set SUPABASE_ANON_KEY env var)",
    )
    parser.add_argument(
        "--session-id", default=None,
        help="GuardEye session UUID (auto-filled when browser connects via --device-id flow)",
    )
    parser.add_argument(
        "--device-id", default=os.environ.get("GUARDEYE_DEVICE_ID", "pi-edge-001"),
        help="Device ID shown in the GuardEye Hardware Connection dialog (default: pi-edge-001)",
    )
    parser.add_argument(
        "--webrtc", action="store_true",
        help="Stream annotated video to GuardEye via WebRTC (requires aiortc + websockets)",
    )

    args = parser.parse_args()

    source_raw = str(args.source)
    source = parse_source(source_raw)

    width = max(160, args.width)
    height = max(120, args.height)
    det_imgsz = max(256, args.det_imgsz)
    pose_imgsz = max(256, args.pose_imgsz)
    det_conf = clamp(args.det_conf, 0.05, 0.95)
    pose_interval = max(1, args.pose_interval)
    contraband_interval = max(1, args.contraband_interval)
    alert_threshold = clamp(args.alert_threshold, 0.01, 1.0)
    alert_cooldown_sec = max(0.0, args.alert_cooldown_sec)
    camera_fps = max(1, args.camera_fps)
    status_every_frames = max(1, args.status_every_frames)
    frame_fit = args.frame_fit
    raspi_mode = bool(args.raspi or detect_raspberry_pi())

    if args.fast:
        width = 512
        height = 384
        det_imgsz = 512
        pose_imgsz = 512
        det_conf = 0.45
        pose_interval = max(2, pose_interval)
        contraband_interval = max(3, contraband_interval)

    if raspi_mode:
        # Camera Module 3 Wide is naturally wide FOV; default to 16:9 output if user kept defaults.
        if args.width == 640 and args.height == 480:
            width = 640
            height = 360
        else:
            width = min(width, 640)
            height = min(height, 480)
        det_imgsz = min(det_imgsz, 512)
        pose_imgsz = min(pose_imgsz, 512)
        pose_interval = max(2, pose_interval)
        contraband_interval = max(2, contraband_interval)
        if args.frame_fit == "stretch":
            frame_fit = "contain"
        # On Pi, only force headless if no graphical display is available.
        if os.environ.get("DISPLAY", "") == "" and not args.show_window:
            args.headless = True

    if args.show_window:
        args.headless = False

    return RuntimeConfig(
        source=source,
        source_raw=source_raw,
        output=args.output,
        width=width,
        height=height,
        det_imgsz=det_imgsz,
        pose_imgsz=pose_imgsz,
        det_conf=det_conf,
        contraband_interval=contraband_interval,
        pose_interval=pose_interval,
        headless=args.headless,
        fast=args.fast,
        report_csv=args.report_csv,
        event_log_csv=args.event_log_csv,
        report_image_dir=args.report_image_dir,
        alert_endpoint=args.alert_endpoint,
        alert_threshold=alert_threshold,
        alert_cooldown_sec=alert_cooldown_sec,
        raspi_mode=raspi_mode,
        camera_fps=camera_fps,
        status_every_frames=status_every_frames,
        frame_fit=frame_fit,
        # --- GuardEye Supabase integration (NEW) ---
        supabase_url=args.supabase_url,
        supabase_anon_key=args.supabase_anon_key,
        session_id=args.session_id,
        webrtc_enabled=bool(args.webrtc),
        device_id=args.device_id,
    )


def fit_frame(frame: np.ndarray, target_w: int, target_h: int, mode: str) -> np.ndarray:
    src_h, src_w = frame.shape[:2]
    if src_w <= 0 or src_h <= 0:
        return frame

    if mode == "stretch":
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    src_aspect = src_w / src_h
    target_aspect = target_w / target_h

    if mode == "contain":
        if src_aspect > target_aspect:
            new_w = target_w
            new_h = int(target_w / src_aspect)
        else:
            new_h = target_h
            new_w = int(target_h * src_aspect)

        resized = cv2.resize(frame, (max(1, new_w), max(1, new_h)), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_off = (target_w - new_w) // 2
        y_off = (target_h - new_h) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
        return canvas

    # cover mode: fill target completely, cropping overflow from center.
    if src_aspect > target_aspect:
        new_h = target_h
        new_w = int(target_h * src_aspect)
    else:
        new_w = target_w
        new_h = int(target_w / src_aspect)

    resized = cv2.resize(frame, (max(1, new_w), max(1, new_h)), interpolation=cv2.INTER_LINEAR)
    x_off = (new_w - target_w) // 2
    y_off = (new_h - target_h) // 2
    return resized[y_off : y_off + target_h, x_off : x_off + target_w]


def print_raspi_camera_diagnostics(source_raw: str) -> None:
    has_libcamera_hello = bool(shutil.which("libcamera-hello"))
    has_rpicam_hello = bool(shutil.which("rpicam-hello"))
    has_v4l2_ctl = bool(shutil.which("v4l2-ctl"))
    has_picamera2 = Picamera2 is not None
    print("Raspberry Pi camera diagnostics:")
    print(f"  source={source_raw}")
    print(f"  DISPLAY={os.environ.get('DISPLAY', '<empty>')}")
    print(f"  libcamera-hello found={has_libcamera_hello}")
    print(f"  rpicam-hello found={has_rpicam_hello}")
    print(f"  v4l2-ctl found={has_v4l2_ctl}")
    print(f"  picamera2 importable={has_picamera2}")
    print(f"  /dev/video0 exists={os.path.exists('/dev/video0')}")
    print(f"  /dev/video1 exists={os.path.exists('/dev/video1')}")
    print("Try these checks on Pi:")
    if has_v4l2_ctl:
        print("  v4l2-ctl --list-devices")
        print("  v4l2-ctl -d /dev/video0 --all")
    if has_libcamera_hello:
        print("  libcamera-hello --list-cameras")
    elif has_rpicam_hello:
        print("  rpicam-hello --list-cameras")
    else:
        print("  libcamera/rpicam hello tools not installed (optional)")
    print("  python proctor_edge.py --source pi --raspi --show-window")
    print("  python proctor_edge.py --source /dev/video0 --raspi --show-window")
    print("  python proctor_edge.py --source /dev/video0 --raspi --headless --output annotated.mp4")


def open_capture_with_backend(source: Union[int, str], backend: Optional[int] = None) -> cv2.VideoCapture:
    if backend is None:
        return cv2.VideoCapture(source)
    return cv2.VideoCapture(source, backend)


def build_raspi_gstreamer_pipelines(width: int, height: int, fps: int) -> List[str]:
    # Try a few common caps chains because OpenCV+GStreamer builds vary on Pi OS images.
    return [
        (
            "libcamerasrc ! "
            f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
            "videoconvert ! appsink drop=true max-buffers=1"
        ),
        (
            "libcamerasrc ! "
            f"video/x-raw,format=NV12,width={width},height={height},framerate={fps}/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"
        ),
        (
            f"v4l2src device=/dev/video0 ! video/x-raw,width={width},height={height},framerate={fps}/1 ! "
            "videoconvert ! appsink drop=true max-buffers=1"
        ),
    ]


def configure_webcam_capture(cap: cv2.VideoCapture, cfg: RuntimeConfig) -> None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(cfg.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(cfg.height))
    cap.set(cv2.CAP_PROP_FPS, float(cfg.camera_fps))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def camera_has_frames(cap: cv2.VideoCapture, attempts: int = 3) -> bool:
    for _ in range(max(1, attempts)):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return True
    return False


def create_capture(source: Union[int, str], cfg: RuntimeConfig) -> Tuple[Optional[cv2.VideoCapture], bool]:
    is_webcam = isinstance(source, int) or is_raspi_camera_source(source)
    source_idx = source if isinstance(source, int) else 0

    if not is_webcam:
        cap = open_capture_with_backend(source)
        if cap.isOpened():
            return cap, False
        cap.release()
        return None, False

    os_name = platform.system().lower()
    candidates: List[Tuple[int, Optional[int]]] = []

    if os_name == "windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
        for b in backends:
            candidates.append((source_idx, b))
        if source_idx == 0:
            for idx in [1, 2]:
                for b in backends:
                    candidates.append((idx, b))
        candidates.append((source_idx, None))
    elif os_name == "linux":
        candidates.append((source_idx, cv2.CAP_V4L2))
        if source_idx == 0:
            candidates.append((1, cv2.CAP_V4L2))
            candidates.append((2, cv2.CAP_V4L2))
        if cfg.raspi_mode:
            if source_idx == 0:
                candidates.append(("/dev/video0", cv2.CAP_V4L2))
                candidates.append(("/dev/video1", cv2.CAP_V4L2))
            if hasattr(cv2, "CAP_GSTREAMER"):
                for pipeline in build_raspi_gstreamer_pipelines(cfg.width, cfg.height, cfg.camera_fps):
                    candidates.append((pipeline, cv2.CAP_GSTREAMER))
        candidates.append((source_idx, None))
    else:
        candidates.append((source_idx, None))
        if source_idx == 0:
            candidates.append((1, None))
            candidates.append((2, None))

    seen = set()
    for idx, backend in candidates:
        key = (idx, backend)
        if key in seen:
            continue
        seen.add(key)
        cap = open_capture_with_backend(idx, backend)
        if cap.isOpened():
            if isinstance(idx, int):
                configure_webcam_capture(cap, cfg)
            if camera_has_frames(cap):
                print(f"Using camera index {idx} backend={backend}")
                return cap, True
            print(f"Camera candidate opened but no frames: source={idx} backend={backend}")
        cap.release()

    if cfg.raspi_mode and (source_idx == 0 or is_raspi_camera_source(source)):
        picam_cap = open_picamera2_capture(cfg)
        if picam_cap is not None:
            return picam_cap, True

    return None, True


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def risk_to_label_color(risk: float) -> Tuple[str, Tuple[int, int, int]]:
    if risk < RISK_NORMAL_MAX:
        return "Normal", COLOR_GREEN
    if risk < RISK_REVIEW_MAX:
        return "Review", COLOR_ORANGE
    return "High Risk", COLOR_RED


def to_int_student_id(track_id: Any) -> Optional[int]:
    try:
        return int(track_id)
    except (TypeError, ValueError):
        return None


def keypoint_xy(conf_kpt: np.ndarray, idx: int, min_conf: float = 0.2) -> Optional[Tuple[float, float]]:
    if idx >= conf_kpt.shape[0]:
        return None
    x, y, c = conf_kpt[idx]
    if c < min_conf:
        return None
    return float(x), float(y)


def infer_head_pose_from_keypoints(kpts: np.ndarray) -> Tuple[str, Tuple[int, int, int], float]:
    nose = keypoint_xy(kpts, 0)
    left_eye = keypoint_xy(kpts, 1)
    right_eye = keypoint_xy(kpts, 2)
    left_ear = keypoint_xy(kpts, 3)
    right_ear = keypoint_xy(kpts, 4)

    if nose is None:
        return "Looking Forward", COLOR_GREEN, 0.0

    anchors = [p for p in [left_eye, right_eye, left_ear, right_ear] if p is not None]
    if anchors:
        avg_anchor_y = sum(p[1] for p in anchors) / len(anchors)
        if nose[1] - avg_anchor_y > 18.0:
            return "Looking Down!", COLOR_ORANGE, HEAD_RISK_DOWN

    horizontal_refs = [p for p in [left_eye, right_eye] if p is not None]
    if len(horizontal_refs) == 2:
        mid_x = (horizontal_refs[0][0] + horizontal_refs[1][0]) / 2.0
        eye_dist = abs(horizontal_refs[0][0] - horizontal_refs[1][0]) + 1e-6
        offset = (nose[0] - mid_x) / eye_dist
        if offset > 0.22:
            return "Looking Right!", COLOR_RED, HEAD_RISK_LEFT_RIGHT
        if offset < -0.22:
            return "Looking Left!", COLOR_RED, HEAD_RISK_LEFT_RIGHT

    if left_eye is not None and right_eye is None and right_ear is not None:
        return "Looking Left!", COLOR_RED, HEAD_RISK_LEFT_RIGHT
    if right_eye is not None and left_eye is None and left_ear is not None:
        return "Looking Right!", COLOR_RED, HEAD_RISK_LEFT_RIGHT

    return "Looking Forward", COLOR_GREEN, 0.0


def now_ms_iso() -> Tuple[int, str]:
    ts_ms = int(time.time() * 1000)
    ts_iso = datetime.now(timezone.utc).isoformat()
    return ts_ms, ts_iso


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def setup_event_log(path: str) -> Optional[EventLogHandle]:
    if path == "":
        return None
    ensure_parent_dir(path)
    file_exists = os.path.exists(path)
    fh = open(path, mode="a", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    if (not file_exists) or os.path.getsize(path) == 0:
        writer.writerow(EVENT_HEADER)
        fh.flush()
    return EventLogHandle(fh=fh, writer=writer)


def close_event_log(handle: Optional[EventLogHandle]) -> None:
    if handle is None:
        return
    handle.fh.close()


def log_event(
    handle: Optional[EventLogHandle],
    frame_index: int,
    student_id: Union[int, str],
    event_type: str,
    risk_score: float,
    head_status: str,
    alert_threshold: float,
    details: str,
) -> None:
    if handle is None:
        return
    ts_ms, ts_iso = now_ms_iso()
    handle.writer.writerow(
        [
            ts_iso,
            ts_ms,
            frame_index,
            student_id,
            event_type,
            f"{risk_score:.4f}",
            head_status,
            f"{alert_threshold:.2f}",
            details,
        ]
    )
    handle.fh.flush()


def post_alert(endpoint: Optional[str], payload: Dict[str, Any]) -> None:
    """Legacy plain-HTTP alert endpoint (kept for backwards compatibility)."""
    if not endpoint:
        return
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=2.5):
            pass
    except (URLError, TimeoutError, OSError) as exc:
        print(f"Alert POST failed: {exc}")


def post_alert_supabase(cfg: "RuntimeConfig", payload: Dict[str, Any]) -> None:
    """
    POST a risk alert directly into the GuardEye Supabase `alerts` table.

    Maps proctor_edge fields → Supabase column names exactly as defined in
    guardeye/supabase_schema.sql so the LiveMonitoring dashboard picks them
    up immediately on its 5-second poll.

    Requires --supabase-url, --supabase-anon-key, and --session-id to be set.
    Falls back silently if any of those are missing.
    """
    if not cfg.supabase_url or not cfg.supabase_anon_key or not cfg.session_id:
        return

    student_id = str(payload.get("student_id", ""))
    row = {
        # user_id is resolved server-side via RLS / service role when using
        # the anon key + a session token.  We omit it here and rely on the
        # Supabase policy that allows insert when session_id matches.
        # If you use a service-role key instead, add "user_id": "<uuid>" here.
        "session_id": cfg.session_id,
        "studentId": student_id,
        "studentName": f"Student {student_id}",
        "behaviorType": payload.get("event_type", "alert_high_risk_triggered"),
        "riskScore": payload.get("risk_score", 0.0),
        "headStatus": payload.get("head_status", None),
        "alertThreshold": payload.get("alert_threshold", cfg.alert_threshold),
        "frameIndex": payload.get("frame_index", None),
        "details": payload.get("details", None),
        "frameUrl": None,  # future: upload snapshot image URL here
    }

    url = f"{cfg.supabase_url.rstrip('/')}/rest/v1/alerts"
    body = json.dumps(row).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "apikey": cfg.supabase_anon_key,
            "Authorization": f"Bearer {cfg.supabase_anon_key}",
            "Prefer": "return=minimal",  # don't return the inserted row (faster)
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=3.0):
            pass
    except (URLError, TimeoutError, OSError) as exc:
        print(f"Supabase alert POST failed: {exc}")


# ---------------------------------------------------------------------------
# WebRTC live video streaming (NEW)
# ---------------------------------------------------------------------------

if AIORTC_AVAILABLE:
    class AnnotatedFrameTrack(VideoStreamTrack):
        """
        An aiortc VideoStreamTrack that pulls annotated BGR frames from a
        thread-safe queue and emits them as H.264-compatible video frames.

        The main capture loop pushes frames into `frame_queue` every iteration.
        The WebRTC event loop reads from it asynchronously via recv().
        """

        kind = "video"

        def __init__(self, frame_queue: "queue.Queue[np.ndarray]", fps: float) -> None:
            super().__init__()
            self._queue = frame_queue
            self._fps = max(1.0, fps)
            self._pts = 0
            self._time_base = fractions.Fraction(1, 90000)

        async def recv(self) -> "av.VideoFrame":
            # Block briefly waiting for a frame; yield to event loop if empty.
            loop = asyncio.get_event_loop()
            bgr: Optional[np.ndarray] = None
            while bgr is None:
                try:
                    bgr = await loop.run_in_executor(None, self._queue.get, True, 0.05)
                except queue.Empty:
                    await asyncio.sleep(0.01)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            pts_increment = int(90000 / self._fps)
            video_frame.pts = self._pts
            video_frame.time_base = self._time_base
            self._pts += pts_increment
            return video_frame


async def _run_webrtc_sender(
    cfg: "RuntimeConfig",
    frame_queue: "queue.Queue[np.ndarray]",
    stop_event: threading.Event,
) -> None:
    """
    Async coroutine that:
      1. Connects to the Supabase Realtime WebSocket for channel `webrtc_{session_id}`.
      2. Waits for an SDP `offer` broadcast from the GuardEye browser.
      3. Creates an RTCPeerConnection, attaches the AnnotatedFrameTrack, and
         sends back an SDP `answer`.
      4. Handles ICE candidate exchange.
      5. Keeps running until `stop_event` is set.

    The channel and message format exactly mirror LiveMonitoring.tsx in guardeye.
    """
    if not cfg.supabase_url or not cfg.supabase_anon_key or not cfg.session_id:
        print("[WebRTC] Missing --supabase-url / --supabase-anon-key / --session-id. Skipping.")
        return

    # Supabase Realtime WebSocket endpoint
    realtime_ws_url = (
        cfg.supabase_url.rstrip("/")
        .replace("https://", "wss://")
        .replace("http://", "ws://")
        + f"/realtime/v1/websocket?apikey={cfg.supabase_anon_key}&vsn=1.0.0"
    )

    channel_topic = f"realtime:webrtc_{cfg.session_id}"
    pc: Optional[RTCPeerConnection] = None
    track: Optional["AnnotatedFrameTrack"] = None
    fps = cfg.camera_fps

    print(f"[WebRTC] Connecting to Supabase Realtime channel webrtc_{cfg.session_id} …")

    try:
        async with websockets.connect(realtime_ws_url) as ws:
            # Join the Realtime channel
            join_msg = json.dumps({
                "topic": channel_topic,
                "event": "phx_join",
                "payload": {},
                "ref": "1",
            })
            await ws.send(join_msg)
            print("[WebRTC] Joined Realtime channel. Waiting for browser offer…")

            async def send_broadcast(event: str, payload: Dict[str, Any]) -> None:
                msg = json.dumps({
                    "topic": channel_topic,
                    "event": "broadcast",
                    "payload": {"type": "broadcast", "event": event, "payload": payload},
                    "ref": None,
                })
                await ws.send(msg)

            async for raw in ws:
                if stop_event.is_set():
                    break

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                event = msg.get("event")
                payload = msg.get("payload", {})

                # Unwrap Supabase broadcast envelope
                if event == "broadcast":
                    inner_event = payload.get("event")
                    inner_payload = payload.get("payload", {})

                    if inner_event == "offer" and pc is None:
                        print("[WebRTC] Received offer from browser. Creating peer connection…")
                        pc = RTCPeerConnection()
                        track = AnnotatedFrameTrack(frame_queue, fps)
                        pc.addTrack(track)

                        @pc.on("icecandidate")
                        async def on_ice(candidate: Any) -> None:
                            if candidate:
                                await send_broadcast("candidate", {"candidate": {
                                    "candidate": candidate.candidate,
                                    "sdpMid": candidate.sdpMid,
                                    "sdpMLineIndex": candidate.sdpMLineIndex,
                                }})

                        offer_sdp = RTCSessionDescription(
                            sdp=inner_payload["offer"]["sdp"],
                            type=inner_payload["offer"]["type"],
                        )
                        await pc.setRemoteDescription(offer_sdp)
                        answer = await pc.createAnswer()
                        await pc.setLocalDescription(answer)
                        await send_broadcast("answer", {"answer": {
                            "sdp": pc.localDescription.sdp,
                            "type": pc.localDescription.type,
                        }})
                        print("[WebRTC] Sent answer. Streaming video…")

                    elif inner_event == "candidate" and pc is not None:
                        cand_data = inner_payload.get("candidate", {})
                        if cand_data.get("candidate"):
                            from aiortc import RTCIceCandidate
                            candidate = RTCIceCandidate(
                                foundation="",
                                component=1,
                                protocol="udp",
                                priority=0,
                                host="",
                                port=0,
                                type="host",
                                sdpMid=cand_data.get("sdpMid"),
                                sdpMLineIndex=cand_data.get("sdpMLineIndex"),
                            )
                            # Pass the raw candidate string via the sdp attribute
                            candidate.sdp = cand_data["candidate"]
                            await pc.addIceCandidate(candidate)

    except Exception as exc:
        print(f"[WebRTC] Connection error: {exc}")
    finally:
        if pc is not None:
            await pc.close()
        print("[WebRTC] Peer connection closed.")


async def _wait_for_session_assignment(cfg: "RuntimeConfig") -> Optional[str]:
    """
    Phase 1 — Boot-and-wait handshake.

    Connects to Supabase Realtime channel `device_cmd_{device_id}` and waits
    indefinitely for the browser to send an `assign-session` event (triggered
    when the teacher types the Device ID and clicks "Connect Device").

    On receiving `assign-session`:
      1. Sends back `device-ack` so the GuardEye UI shows "Device connected!"
      2. Returns the session UUID for Phase 2.

    If cfg.session_id is already set (manual --session-id flag), skips Phase 1
    and returns immediately.
    """
    if cfg.session_id:
        print(f"[Device] Session ID pre-set: {cfg.session_id}. Skipping handshake.")
        return cfg.session_id

    if not cfg.supabase_url or not cfg.supabase_anon_key:
        print("[Device] Missing --supabase-url / --supabase-anon-key. Cannot listen for assignment.")
        return None

    realtime_ws_url = (
        cfg.supabase_url.rstrip("/")
        .replace("https://", "wss://")
        .replace("http://", "ws://")
        + f"/realtime/v1/websocket?apikey={cfg.supabase_anon_key}&vsn=1.0.0"
    )
    cmd_channel = f"realtime:device_cmd_{cfg.device_id}"

    print(f"[Device] Listening on channel device_cmd_{cfg.device_id}")
    print(f"[Device] >>> Enter Device ID  \"{cfg.device_id}\"  in the GuardEye Hardware Connection dialog <<<")

    while True:  # reconnect loop
        try:
            async with websockets.connect(realtime_ws_url, ping_interval=30) as ws:
                await ws.send(json.dumps({
                    "topic": cmd_channel,
                    "event": "phx_join",
                    "payload": {},
                    "ref": "1",
                }))

                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    payload = msg.get("payload", {})
                    if msg.get("event") == "broadcast":
                        inner_event = payload.get("event")
                        inner_payload = payload.get("payload", {})

                        if inner_event == "assign-session":
                            session_id = inner_payload.get("sessionId")
                            if not session_id:
                                continue

                            print(f"[Device] Assigned to session: {session_id}")

                            # Send device-ack so the GuardEye UI shows "Device connected!"
                            await ws.send(json.dumps({
                                "topic": cmd_channel,
                                "event": "broadcast",
                                "payload": {
                                    "type": "broadcast",
                                    "event": "device-ack",
                                    "payload": {},
                                },
                                "ref": None,
                            }))
                            print("[Device] Sent device-ack. GuardEye UI should show 'Device connected!'")
                            return session_id

        except Exception as exc:
            print(f"[Device] Realtime error: {exc}. Reconnecting in 5 s…")
            await asyncio.sleep(5)


async def _run_guardeye_flow(
    cfg: "RuntimeConfig",
    frame_queue: "queue.Queue[np.ndarray]",
    stop_event: threading.Event,
) -> None:
    """
    Full GuardEye connection flow:
      Phase 1 — wait for browser to assign a session via Hardware Connection dialog.
      Phase 2 — start WebRTC video streaming for that session.

    Loops back to Phase 1 after each session ends so the Pi is always ready
    for the next exam without restarting.
    """
    while not stop_event.is_set():
        # Phase 1: wait for session assignment
        session_id = await _wait_for_session_assignment(cfg)
        if not session_id:
            print("[Device] No session ID obtained. Retrying in 10 s…")
            await asyncio.sleep(10)
            continue

        # Inject session_id into cfg for alert posting and WebRTC channel
        cfg.session_id = session_id

        # Phase 2: WebRTC video streaming
        await _run_webrtc_sender(cfg, frame_queue, stop_event)

        # After session ends, clear session_id and go back to listening
        print("[Device] Session ended. Returning to standby — ready for next connection.")
        cfg.session_id = None


def start_webrtc_thread(
    cfg: "RuntimeConfig",
    frame_queue: "queue.Queue[np.ndarray]",
    stop_event: threading.Event,
) -> Optional[threading.Thread]:
    """
    Starts the full GuardEye connection flow in a dedicated asyncio thread:

      Phase 1 — Device handshake (if session_id not already set):
        Listens on Supabase Realtime channel `device_cmd_{device_id}`.
        When the browser sends `assign-session`, replies with `device-ack`
        and captures the session UUID.

      Phase 2 — WebRTC streaming:
        Joins channel `webrtc_{session_id}`, waits for SDP offer from the
        browser, replies with answer, streams annotated frames.

    Returns the thread so main() can join it on exit.
    """
    if not AIORTC_AVAILABLE:
        print("[WebRTC] aiortc not installed — live video streaming disabled.")
        print("         Install with:  pip install aiortc websockets")
        return None
    if not WEBSOCKETS_AVAILABLE:
        print("[WebRTC] websockets not installed — live video streaming disabled.")
        print("         Install with:  pip install websockets")
        return None

    def _thread_target() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_run_guardeye_flow(cfg, frame_queue, stop_event))
        loop.close()

    t = threading.Thread(target=_thread_target, daemon=True, name="webrtc-sender")
    t.start()
    return t


def xyxy_to_ltwh(xyxy: Tuple[float, float, float, float]) -> List[float]:
    x1, y1, x2, y2 = xyxy
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def center_of_box(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def clamp_box_to_frame(box: Tuple[float, float, float, float], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    ix1 = int(max(0, min(w - 1, x1)))
    iy1 = int(max(0, min(h - 1, y1)))
    ix2 = int(max(0, min(w - 1, x2)))
    iy2 = int(max(0, min(h - 1, y2)))
    if ix2 <= ix1:
        ix2 = min(w - 1, ix1 + 1)
    if iy2 <= iy1:
        iy2 = min(h - 1, iy1 + 1)
    return ix1, iy1, ix2, iy2


def detect_people_and_contraband(
    det_model: YOLO,
    frame: np.ndarray,
    det_imgsz: int,
    det_conf: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    results = det_model(frame, imgsz=det_imgsz, conf=det_conf, verbose=False)
    result = results[0]
    names = result.names

    persons: List[Dict[str, Any]] = []
    contraband: List[Dict[str, Any]] = []

    boxes = result.boxes
    if boxes is None:
        return persons, contraband

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        box_xyxy = (float(x1), float(y1), float(x2), float(y2))
        name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

        if cls_id == PERSON_CLASS_ID:
            persons.append(
                {
                    "box": box_xyxy,
                    "conf": conf,
                    "class_id": cls_id,
                    "name": "person",
                }
            )
        elif cls_id in CONTRABAND_CLASS_IDS and name in CONTRABAND_WEIGHTS:
            contraband.append(
                {
                    "box": box_xyxy,
                    "conf": conf,
                    "class_id": cls_id,
                    "name": name,
                }
            )

    return persons, contraband


def detect_pose(
    pose_model: YOLO,
    frame: np.ndarray,
    pose_imgsz: int,
) -> List[Dict[str, Any]]:
    results = pose_model(frame, imgsz=pose_imgsz, verbose=False)
    result = results[0]

    if result.boxes is None or result.keypoints is None:
        return []

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    kpts_data = result.keypoints.data.cpu().numpy()

    outputs: List[Dict[str, Any]] = []
    for i in range(min(len(boxes_xyxy), len(kpts_data))):
        x1, y1, x2, y2 = boxes_xyxy[i].tolist()
        outputs.append(
            {
                "box": (float(x1), float(y1), float(x2), float(y2)),
                "kpts": kpts_data[i],
            }
        )
    return outputs


def match_pose_to_tracks(
    tracked_students: Dict[Union[int, str], Dict[str, Any]],
    pose_dets: List[Dict[str, Any]],
) -> Tuple[Dict[Union[int, str], Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    assigned_pose: Dict[Union[int, str], Dict[str, Any]] = {}
    unmatched_pose: Dict[str, Dict[str, Any]] = {}

    used_students: set = set()
    temp_counter = 1

    for pose in pose_dets:
        pbox = pose["box"]
        best_sid = None
        best_iou = 0.0
        for sid, info in tracked_students.items():
            if sid in used_students:
                continue
            iou = iou_xyxy(pbox, info["box"])
            if iou > best_iou:
                best_iou = iou
                best_sid = sid

        if best_sid is not None and best_iou > 0.1:
            assigned_pose[best_sid] = pose
            used_students.add(best_sid)
        else:
            unmatched_pose[f"pose-{temp_counter}"] = pose
            temp_counter += 1

    return assigned_pose, unmatched_pose


def process_frame(
    frame: np.ndarray,
    frame_index: int,
    det_model: YOLO,
    pose_model: YOLO,
    tracker: DeepSort,
    cfg: RuntimeConfig,
    pose_cache: List[Dict[str, Any]],
    contraband_cache: List[Dict[str, Any]],
    student_history: Dict[int, Dict[str, Any]],
) -> Tuple[
    np.ndarray,
    Dict[Union[int, str], Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    persons, det_contraband = detect_people_and_contraband(
        det_model, frame, cfg.det_imgsz, cfg.det_conf
    )

    if frame_index % cfg.contraband_interval == 0:
        contraband_cache = det_contraband

    if frame_index % cfg.pose_interval == 0:
        pose_cache = detect_pose(pose_model, frame, cfg.pose_imgsz)

    detections_for_tracker = []
    for p in persons:
        detections_for_tracker.append((xyxy_to_ltwh(p["box"]), p["conf"], "person"))

    tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

    tracked_students: Dict[Union[int, str], Dict[str, Any]] = {}
    for track in tracks:
        if not track.is_confirmed():
            continue
        if track.time_since_update > 1:
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        box = (float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3]))
        center = center_of_box(box)

        tracked_students[track_id] = {
            "track_id": track_id,
            "box": box,
            "center": center,
            "head_status": "Looking Forward",
            "head_risk": 0.0,
            "motion_risk": 0.0,
            "proximity_risk": 0.0,
            "contraband_risk": 0.0,
            "risk": 0.0,
            "label": "Normal",
            "color": COLOR_GREEN,
            "nose": None,
        }

    assigned_pose, unmatched_pose = match_pose_to_tracks(tracked_students, pose_cache)

    for sid, pose in assigned_pose.items():
        status, _, head_risk = infer_head_pose_from_keypoints(pose["kpts"])
        tracked_students[sid]["head_status"] = status
        tracked_students[sid]["head_risk"] = head_risk
        tracked_students[sid]["nose"] = keypoint_xy(pose["kpts"], 0)

    for _pid, pose in unmatched_pose.items():
        # Keep unmatched pose structure for deterministic behavior and debugging.
        _ = pose

    for sid, info in tracked_students.items():
        int_sid = to_int_student_id(sid)
        if int_sid is None:
            continue
        hist = student_history.setdefault(int_sid, {})
        if "last_center" in hist:
            prev_center = hist["last_center"]
            movement = math.dist(info["center"], prev_center)
            if movement > MOTION_THRESHOLD:
                info["motion_risk"] = clamp((movement / MOTION_THRESHOLD) * 0.10, 0.0, 0.25)
        hist["last_center"] = info["center"]

    int_student_ids = [sid for sid in tracked_students.keys() if to_int_student_id(sid) is not None]
    for i in range(len(int_student_ids)):
        sid_a = int_student_ids[i]
        for j in range(i + 1, len(int_student_ids)):
            sid_b = int_student_ids[j]
            ca = tracked_students[sid_a]["center"]
            cb = tracked_students[sid_b]["center"]
            dist = math.dist(ca, cb)
            if dist < PROXIMITY_THRESHOLD:
                prox = clamp(((PROXIMITY_THRESHOLD - dist) / PROXIMITY_THRESHOLD) * 0.20, 0.0, 0.20)
                tracked_students[sid_a]["proximity_risk"] = max(tracked_students[sid_a]["proximity_risk"], prox)
                tracked_students[sid_b]["proximity_risk"] = max(tracked_students[sid_b]["proximity_risk"], prox)

    global_contraband_alerts: List[Dict[str, Any]] = []
    for item in contraband_cache:
        nearest_sid = None
        nearest_dist = float("inf")
        item_center = center_of_box(item["box"])

        for sid, info in tracked_students.items():
            dist = math.dist(item_center, info["center"])
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_sid = sid

        if nearest_sid is not None and nearest_dist <= CONTRABAND_ASSOCIATION_THRESHOLD:
            weight = CONTRABAND_WEIGHTS.get(item["name"], 0.0)
            tracked_students[nearest_sid]["contraband_risk"] += weight
        else:
            global_contraband_alerts.append(
                {
                    "name": item["name"],
                    "conf": item["conf"],
                    "box": item["box"],
                    "distance": nearest_dist,
                }
            )

    for sid, info in tracked_students.items():
        total_risk = (
            info["head_risk"]
            + info["motion_risk"]
            + info["proximity_risk"]
            + info["contraband_risk"]
        )
        info["risk"] = clamp(total_risk, 0.0, 1.0)
        label, color = risk_to_label_color(info["risk"])
        info["label"] = label
        info["color"] = color

    annotated = frame.copy()

    for sid, info in tracked_students.items():
        x1, y1, x2, y2 = [int(v) for v in info["box"]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), info["color"], 2)

        text = f"Student {sid}: {info['head_status']} | {info['label']} {info['risk']:.2f}"
        cv2.putText(
            annotated,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            info["color"],
            2,
            cv2.LINE_AA,
        )

        if info["nose"] is not None:
            nx, ny = int(info["nose"][0]), int(info["nose"][1])
            cv2.circle(annotated, (nx, ny), 4, COLOR_CYAN, -1)

    for item in contraband_cache:
        x1, y1, x2, y2 = [int(v) for v in item["box"]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_YELLOW, 2)
        label = f"{item['name']} {item['conf']:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_YELLOW,
            2,
            cv2.LINE_AA,
        )

    return annotated, tracked_students, pose_cache, contraband_cache, global_contraband_alerts


def update_student_report_stats(
    tracked_students: Dict[Union[int, str], Dict[str, Any]],
    student_history: Dict[int, Dict[str, Any]],
) -> None:
    for sid, info in tracked_students.items():
        int_sid = to_int_student_id(sid)
        if int_sid is None:
            continue
        hist = student_history.setdefault(int_sid, {})
        hist["samples"] = int(hist.get("samples", 0)) + 1
        hist["risk_sum"] = float(hist.get("risk_sum", 0.0)) + float(info["risk"])
        hist["risk_max"] = max(float(hist.get("risk_max", 0.0)), float(info["risk"]))
        hist["final_label"] = info["label"]
        hist.setdefault("best_image_risk", -1.0)
        hist.setdefault("image_path", "")
        hist.setdefault("image_captured_at_ms", "")
        hist.setdefault("last_head_status", "Looking Forward")
        hist.setdefault("last_contraband_risk", 0.0)
        hist.setdefault("above_threshold", False)
        hist.setdefault("last_alert_ms", -10**12)


def maybe_capture_snapshot(
    frame: np.ndarray,
    tracked_students: Dict[Union[int, str], Dict[str, Any]],
    student_history: Dict[int, Dict[str, Any]],
    report_image_dir: str,
) -> None:
    os.makedirs(report_image_dir, exist_ok=True)
    frame_h, frame_w = frame.shape[:2]

    for sid, info in tracked_students.items():
        int_sid = to_int_student_id(sid)
        if int_sid is None:
            continue

        hist = student_history.setdefault(int_sid, {})
        current_risk = float(info["risk"])
        best_risk = float(hist.get("best_image_risk", -1.0))

        if current_risk <= best_risk:
            continue

        x1, y1, x2, y2 = clamp_box_to_frame(info["box"], frame_w, frame_h)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        image_path = os.path.join(report_image_dir, f"student_{int_sid}.jpg")
        saved = cv2.imwrite(image_path, crop)
        if not saved:
            continue

        ts_ms, _ = now_ms_iso()
        hist["best_image_risk"] = current_risk
        hist["image_path"] = image_path
        hist["image_captured_at_ms"] = ts_ms


def handle_events_and_alerts(
    frame_index: int,
    tracked_students: Dict[Union[int, str], Dict[str, Any]],
    student_history: Dict[int, Dict[str, Any]],
    event_log: Optional[EventLogHandle],
    cfg: RuntimeConfig,
) -> None:
    for sid, info in tracked_students.items():
        int_sid = to_int_student_id(sid)
        if int_sid is None:
            continue

        hist = student_history.setdefault(int_sid, {})
        previous_head = str(hist.get("last_head_status", "Looking Forward"))
        current_head = str(info["head_status"])

        if previous_head == "Looking Forward" and current_head != "Looking Forward":
            log_event(
                event_log,
                frame_index,
                int_sid,
                "head_pose_warning",
                info["risk"],
                current_head,
                cfg.alert_threshold,
                "transition=forward_to_non_forward",
            )

        previous_contraband = float(hist.get("last_contraband_risk", 0.0))
        current_contraband = float(info["contraband_risk"])
        if current_contraband > previous_contraband + 1e-9:
            log_event(
                event_log,
                frame_index,
                int_sid,
                "contraband_associated",
                info["risk"],
                current_head,
                cfg.alert_threshold,
                f"contraband_risk={current_contraband:.2f}",
            )

        risk = float(info["risk"])
        above_threshold = bool(hist.get("above_threshold", False))
        ts_ms, ts_iso = now_ms_iso()

        if (not above_threshold) and risk >= cfg.alert_threshold:
            last_alert_ms = int(hist.get("last_alert_ms", -10**12))
            cooldown_ms = int(cfg.alert_cooldown_sec * 1000)
            cooldown_ok = (ts_ms - last_alert_ms) >= cooldown_ms

            if cooldown_ok:
                log_event(
                    event_log,
                    frame_index,
                    int_sid,
                    "alert_high_risk_triggered",
                    risk,
                    current_head,
                    cfg.alert_threshold,
                    "threshold_crossed=true",
                )
                payload = {
                    "event_type": "alert_high_risk_triggered",
                    "student_id": int_sid,
                    "risk_score": round(risk, 4),
                    "alert_threshold": cfg.alert_threshold,
                    "frame_index": frame_index,
                    "timestamp_ms": ts_ms,
                    "timestamp_iso": ts_iso,
                    "source": cfg.source_raw,
                    # NEW: include head_status and details for Supabase mapping
                    "head_status": current_head,
                    "details": f"head={current_head} contraband={info.get('contraband_risk', 0.0):.2f}",
                }
                post_alert(cfg.alert_endpoint, payload)
                post_alert_supabase(cfg, payload)  # NEW: direct Supabase insert
                hist["last_alert_ms"] = ts_ms

            hist["above_threshold"] = True

        elif above_threshold and risk < cfg.alert_threshold:
            log_event(
                event_log,
                frame_index,
                int_sid,
                "alert_high_risk_cleared",
                risk,
                current_head,
                cfg.alert_threshold,
                "threshold_cleared=true",
            )
            hist["above_threshold"] = False

        hist["last_head_status"] = current_head
        hist["last_contraband_risk"] = current_contraband


def print_exam_summary(student_history: Dict[int, Dict[str, Any]]) -> None:
    print("\nExam Summary")
    print("=" * 60)
    if not student_history:
        print("No students tracked.")
        return

    for sid in sorted(student_history.keys()):
        hist = student_history[sid]
        samples = int(hist.get("samples", 0))
        avg_risk = (float(hist.get("risk_sum", 0.0)) / samples) if samples > 0 else 0.0
        max_risk = float(hist.get("risk_max", 0.0))
        final_label = str(hist.get("final_label", "Normal"))
        print(
            f"Student {sid}: samples={samples}, avg_risk={avg_risk:.3f}, "
            f"max_risk={max_risk:.3f}, final_label={final_label}"
        )


def write_report_csv(path: str, student_history: Dict[int, Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, mode="w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(REPORT_HEADER)

        for sid in sorted(student_history.keys()):
            hist = student_history[sid]
            samples = int(hist.get("samples", 0))
            avg_risk = (float(hist.get("risk_sum", 0.0)) / samples) if samples > 0 else 0.0
            max_risk = float(hist.get("risk_max", 0.0))
            final_label = str(hist.get("final_label", "Normal"))
            image_path = str(hist.get("image_path", ""))
            image_captured_at_ms = hist.get("image_captured_at_ms", "")

            writer.writerow(
                [
                    sid,
                    samples,
                    f"{avg_risk:.4f}",
                    f"{max_risk:.4f}",
                    final_label,
                    image_path,
                    image_captured_at_ms,
                ]
            )


def main() -> None:
    cfg = parse_args()

    if cfg.raspi_mode:
        print("Raspberry Pi mode enabled")
        print(
            f"Pi runtime config: {cfg.width}x{cfg.height}, det_imgsz={cfg.det_imgsz}, "
            f"pose_imgsz={cfg.pose_imgsz}, pose_interval={cfg.pose_interval}, "
            f"contraband_interval={cfg.contraband_interval}, headless={cfg.headless}, "
            f"frame_fit={cfg.frame_fit}"
        )

    if torch is not None:
        try:
            thread_count = max(1, (os.cpu_count() or 2) // 2)
            torch.set_num_threads(thread_count)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

    pose_model = YOLO("yolov8n-pose.pt")
    det_model = YOLO("yolov8n.pt")

    tracker = DeepSort(
        max_age=30,
        n_init=2,
        nms_max_overlap=1.0,
        max_cosine_distance=0.3,
    )

    cap, is_webcam = create_capture(cfg.source, cfg)
    if cap is None:
        print(f"Unable to open source: {cfg.source_raw}")
        if cfg.raspi_mode:
            print("Raspberry Pi camera hint: try --source pi --raspi --camera-fps 30")
            print("If running over SSH without display, use --headless and provide --output annotated.mp4")
            print_raspi_camera_diagnostics(cfg.source_raw)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0

    writer: Optional[cv2.VideoWriter] = None
    if cfg.output:
        ensure_parent_dir(cfg.output)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(cfg.output, fourcc, fps, (cfg.width, cfg.height))

    event_log = setup_event_log(cfg.event_log_csv)

    frame_index = 0
    pose_cache: List[Dict[str, Any]] = []
    contraband_cache: List[Dict[str, Any]] = []
    student_history: Dict[int, Dict[str, Any]] = {}

    # --- WebRTC live video (NEW) ---
    webrtc_stop = threading.Event()
    frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2)  # 2-frame buffer
    webrtc_thread: Optional[threading.Thread] = None

    # Auto-enable GuardEye integration when Supabase creds are present,
    # even if --webrtc flag was not explicitly passed.
    guardeye_active = cfg.webrtc_enabled or bool(cfg.supabase_url and cfg.supabase_anon_key)

    if guardeye_active:
        print("")
        print("=" * 55)
        print(f"  GuardEye Device ID  :  {cfg.device_id}")
        print(f"  Go to               :  https://guardeye.onrender.com/dashboard/start")
        print(f"  Select a session, click 'Start', enter the Device ID above")
        print("=" * 55)
        print("")
        cfg.webrtc_enabled = True  # ensure frame pushing is active
        webrtc_thread = start_webrtc_thread(cfg, frame_queue, webrtc_stop)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame = fit_frame(frame, cfg.width, cfg.height, cfg.frame_fit)
            if is_webcam:
                frame = cv2.flip(frame, 1)

            annotated, tracked_students, pose_cache, contraband_cache, _global_items = process_frame(
                frame,
                frame_index,
                det_model,
                pose_model,
                tracker,
                cfg,
                pose_cache,
                contraband_cache,
                student_history,
            )

            update_student_report_stats(tracked_students, student_history)
            maybe_capture_snapshot(frame, tracked_students, student_history, cfg.report_image_dir)
            handle_events_and_alerts(frame_index, tracked_students, student_history, event_log, cfg)

            # --- Push annotated frame to WebRTC sender (NEW) ---
            if cfg.webrtc_enabled and webrtc_thread is not None:
                try:
                    # Non-blocking put; drop frame if consumer is behind (avoids blocking capture)
                    frame_queue.put_nowait(annotated.copy())
                except queue.Full:
                    pass

            if cfg.headless and (frame_index % cfg.status_every_frames == 0):
                student_count = len(tracked_students)
                max_risk = max((float(s["risk"]) for s in tracked_students.values()), default=0.0)
                print(
                    f"status frame={frame_index} tracked_students={student_count} "
                    f"max_risk={max_risk:.2f}"
                )

            if writer is not None:
                writer.write(annotated)

            if not cfg.headless:
                cv2.imshow("AI Proctor", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        close_event_log(event_log)
        cv2.destroyAllWindows()
        # --- Stop WebRTC thread (NEW) ---
        if cfg.webrtc_enabled:
            webrtc_stop.set()
            if webrtc_thread is not None:
                webrtc_thread.join(timeout=5.0)

    print_exam_summary(student_history)
    write_report_csv(cfg.report_csv, student_history)
    print(f"\nReport CSV written: {cfg.report_csv}")


if __name__ == "__main__":
    main()
