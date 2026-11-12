$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Lock = "configs/dependencies/windows-py312-cu130.lock.txt"
$VenvPython = ".\.venv\Scripts\python.exe"

if (!(Test-Path $Lock)) {
    throw "Dependency lock not found"
}

Write-Host "COMPANY VISION - WINDOWS SETUP" -ForegroundColor Cyan

# Python 3.12 is required by the project.
& py -3.12 -c "import sys; assert sys.version_info[:2] == (3, 12)"
if ($LASTEXITCODE -ne 0) {
    throw "Install Python 3.12 with the Windows py launcher"
}

if (!(Test-Path $VenvPython)) {
    & py -3.12 -m venv .venv

    if ($LASTEXITCODE -ne 0) {
        throw "Virtual environment creation failed"
    }
}

& $VenvPython -m pip install -r $Lock
if ($LASTEXITCODE -ne 0) {
    throw "Locked dependency installation failed"
}

& $VenvPython -m pip install --no-deps -e .
if ($LASTEXITCODE -ne 0) {
    throw "Local project installation failed"
}

& $VenvPython -m pip check
if ($LASTEXITCODE -ne 0) {
    throw "Dependency verification failed"
}

& $VenvPython -m pytest -q
if ($LASTEXITCODE -ne 0) {
    throw "Project tests failed"
}

Write-Host "`nGPU STATUS:" -ForegroundColor Cyan

& $VenvPython -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"

if ($LASTEXITCODE -ne 0) {
    throw "PyTorch verification failed"
}

Write-Host "`nDAHUA SDK STATUS:" -ForegroundColor Cyan

& $VenvPython -c "import importlib.util; print('NetSDK installed:', importlib.util.find_spec('NetSDK') is not None)"

Write-Host "`nSETUP COMPLETED" -ForegroundColor Green
Write-Host "Dahua vendor DLL and YOLO model weights must be supplied separately if absent."