# Company Vision - Second System Setup

## Requirements

- Windows 10/11 x64
- Python 3.12 x64
- Git
- NVIDIA driver compatible with CUDA 13.0
- Compatible NVIDIA GPU for CUDA inference

## Clone

```powershell
New-Item -ItemType Directory -Force D:\Projects | Out-Null
Set-Location D:\Projects

git clone --branch feature/phase4-node4-5-async-main --single-branch https://github.com/ahanmadan/company-vision.git company-vision

Set-Location D:\Projects\company-vision
```

## Install

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
```

## Dahua SDK

The official Dahua NetSDK wheel and native DLL files are not stored in Git.

Transfer the compatible official SDK distribution to the second system.

```powershell
.\scripts\setup_windows.ps1 -DahuaWheelPath "D:\SDK\NetSDK-2.0.0.1-py3-none-win_amd64.whl"
```

The SDK native DLL files must also be available according to the official distribution instructions.

## YOLO Model

The model weights are not committed to Git.

Obtain yolo26x-pose.pt from an authorized source.

## Original Node 4.4 Runner

```powershell
.\.venv\Scripts\python.exe scripts\run_dahua_live_pose.py
```

## Independent Node 4.5 Runner

```powershell
.\.venv\Scripts\python.exe scripts\run_dahua_live_pose_node45.py --async-inference
```

## Benchmark

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_node45_live.py --duration 30 --repeats 2 --warmup-runs 5
```

## Development Rules

- Always use the repository root.
- Never copy the virtual environment between computers.
- Each computer creates its own .venv.
- Never commit .env or camera credentials.
- Never commit model weights or surveillance recordings.
- Keep the Python version and locked dependencies consistent.
- Preserve approved Node 4.4 modules.
- Do not modify previously approved node contracts without an explicit decision.
