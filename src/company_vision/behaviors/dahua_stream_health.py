"""Node 5.10.1 - Real Dahua NetSDK stream health validation."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


class StreamHealthError(RuntimeError):
    """Native video acquisition or decoding failed."""


@dataclass(frozen=True, slots=True)
class StreamHealthReport:
    status: str
    transport: str
    channel: int
    stream_type: int
    duration_seconds: int
    recorded_bytes: int
    decoded_frame: bool
    width: int
    height: int


def validate_recording(
    path: Path,
    minimum_bytes: int = 32768,
) -> int:
    if minimum_bytes < 1:
        raise ValueError("minimum_bytes must be positive")

    if not path.is_file():
        raise StreamHealthError("Recording was not created")

    size = path.stat().st_size

    if size < minimum_bytes:
        raise StreamHealthError(
            f"Insufficient stream data: {size} bytes"
        )

    return size


def validate_dimensions(width: int, height: int) -> None:
    if width < 64 or height < 64:
        raise StreamHealthError("Invalid decoded dimensions")


def sdk_failure(sdk, operation: str) -> StreamHealthError:
    return StreamHealthError(
        f"{operation} failed, error={sdk.CLIENT_GetLastError()}"
    )


def record_netsdk(
    *,
    dll_path: Path,
    host: str,
    port: int,
    username: str,
    password: str,
    channel: int,
    stream_type: int,
    seconds: int,
    destination: Path,
) -> int:
    if sys.platform != "win32":
        raise StreamHealthError("Windows is required")

    if not dll_path.is_file():
        raise StreamHealthError("NetSDK DLL missing")

    if not host or not username or not password:
        raise StreamHealthError("Camera credentials missing")

    if not 1 <= port <= 65535:
        raise StreamHealthError("Invalid port")

    if channel < 0 or stream_type not in (0, 1):
        raise StreamHealthError("Invalid channel or stream type")

    if not 3 <= seconds <= 30:
        raise StreamHealthError("Invalid recording duration")

    destination.parent.mkdir(parents=True, exist_ok=True)

    dll_scope = os.add_dll_directory(str(dll_path.parent))
    sdk = None
    initialized = False
    login = 0
    real = 0
    saving = False

    try:
        sdk = ctypes.WinDLL(str(dll_path))

        handle_type = ctypes.c_longlong
        sdk.CLIENT_Init.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        sdk.CLIENT_Init.restype = ctypes.c_int

        sdk.CLIENT_Cleanup.argtypes = []
        sdk.CLIENT_Cleanup.restype = None

        sdk.CLIENT_GetLastError.argtypes = []
        sdk.CLIENT_GetLastError.restype = ctypes.c_uint

        sdk.CLIENT_LoginEx2.argtypes = [
            ctypes.c_char_p,
            ctypes.c_ushort,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        sdk.CLIENT_LoginEx2.restype = handle_type

        sdk.CLIENT_Logout.argtypes = [handle_type]
        sdk.CLIENT_Logout.restype = ctypes.c_int

        sdk.CLIENT_RealPlayEx.argtypes = [
            handle_type,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        sdk.CLIENT_RealPlayEx.restype = handle_type

        sdk.CLIENT_StopRealPlayEx.argtypes = [handle_type]
        sdk.CLIENT_StopRealPlayEx.restype = ctypes.c_int

        sdk.CLIENT_SaveRealData.argtypes = [
            handle_type,
            ctypes.c_char_p,
        ]
        sdk.CLIENT_SaveRealData.restype = ctypes.c_int

        sdk.CLIENT_StopSaveRealData.argtypes = [handle_type]
        sdk.CLIENT_StopSaveRealData.restype = ctypes.c_int

        if not sdk.CLIENT_Init(None, None):
            raise sdk_failure(sdk, "CLIENT_Init")

        initialized = True

        device_info = ctypes.create_string_buffer(4096)
        error_code = ctypes.c_int()

        login = int(sdk.CLIENT_LoginEx2(
            host.encode("utf-8"),
            port,
            username.encode("utf-8"),
            password.encode("utf-8"),
            0,
            None,
            ctypes.byref(device_info),
            ctypes.byref(error_code),
        ))

        if not login:
            raise sdk_failure(sdk, "CLIENT_LoginEx2")

        real = int(sdk.CLIENT_RealPlayEx(
            login,
            channel,
            None,
            stream_type,
        ))

        if not real:
            raise sdk_failure(sdk, "CLIENT_RealPlayEx")

        destination_bytes = str(
            destination.resolve()
        ).encode("mbcs")

        if not sdk.CLIENT_SaveRealData(
            real,
            destination_bytes,
        ):
            raise sdk_failure(sdk, "CLIENT_SaveRealData")

        saving = True
        time.sleep(seconds)

        if not sdk.CLIENT_StopSaveRealData(real):
            raise sdk_failure(sdk, "CLIENT_StopSaveRealData")

        saving = False

        return validate_recording(destination)

    finally:
        if sdk is not None:
            if saving and real:
                sdk.CLIENT_StopSaveRealData(real)

            if real:
                sdk.CLIENT_StopRealPlayEx(real)

            if login:
                sdk.CLIENT_Logout(login)

            if initialized:
                sdk.CLIENT_Cleanup()

        dll_scope.close()


def decode_frame(
    recording: Path,
    destination: Path,
) -> tuple[int, int]:
    if shutil.which("ffmpeg") is None:
        raise StreamHealthError("FFmpeg not found in PATH")

    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-i", str(recording),
                "-frames:v", "1",
                "-update", "1",
                str(destination),
            ],
            check=True,
            capture_output=True,
            timeout=45,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise StreamHealthError(
            "Stream decoding failed"
        ) from exc

    if not destination.is_file():
        raise StreamHealthError("Decoded frame missing")

    try:
        import cv2
    except ImportError as exc:
        raise StreamHealthError("OpenCV missing") from exc

    frame = cv2.imread(str(destination))

    if frame is None:
        raise StreamHealthError("Decoded frame unreadable")

    height, width = frame.shape[:2]
    validate_dimensions(width, height)

    return width, height


def run_health_check(
    *,
    dll_path: Path,
    host: str,
    port: int,
    username: str,
    password: str,
    channel: int,
    stream_type: int,
    seconds: int,
    output: Path,
) -> StreamHealthReport:
    output.mkdir(parents=True, exist_ok=True)

    recording = output / "netsdk_sample.dav"
    image = output / "netsdk_frame.png"

    for path in (recording, image):
        if path.exists():
            path.unlink()

    byte_count = record_netsdk(
        dll_path=dll_path,
        host=host,
        port=port,
        username=username,
        password=password,
        channel=channel,
        stream_type=stream_type,
        seconds=seconds,
        destination=recording,
    )

    width, height = decode_frame(recording, image)

    report = StreamHealthReport(
        status="pass",
        transport="Dahua NetSDK",
        channel=channel,
        stream_type=stream_type,
        duration_seconds=seconds,
        recorded_bytes=byte_count,
        decoded_frame=True,
        width=width,
        height=height,
    )

    (output / "health_report.json").write_text(
        json.dumps(asdict(report), indent=2),
        encoding="utf-8",
    )

    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--stream-type", type=int, default=0)
    parser.add_argument("--seconds", type=int, default=8)
    parser.add_argument(
        "--output",
        default="artifacts/local_validation/node_5_10_1",
    )
    args = parser.parse_args()

    if not args.live:
        parser.error("--live is required")

    required = (
        "CV_NETSDK_DLL",
        "CV_DAHUA_HOST",
        "CV_DAHUA_USER",
        "CV_DAHUA_PASSWORD",
    )

    missing = [x for x in required if not os.getenv(x)]

    if missing:
        raise StreamHealthError(
            "Missing configuration: " + ", ".join(missing)
        )

    report = run_health_check(
        dll_path=Path(os.environ["CV_NETSDK_DLL"]),
        host=os.environ["CV_DAHUA_HOST"],
        port=int(os.getenv("CV_DAHUA_PORT", "37777")),
        username=os.environ["CV_DAHUA_USER"],
        password=os.environ["CV_DAHUA_PASSWORD"],
        channel=args.channel,
        stream_type=args.stream_type,
        seconds=args.seconds,
        output=Path(args.output),
    )

    print("REAL NETSDK STREAM: PASS")
    print(f"RECORDED BYTES: {report.recorded_bytes}")
    print(f"FRAME DIMENSIONS: {report.width}x{report.height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())