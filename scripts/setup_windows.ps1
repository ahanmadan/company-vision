[CmdletBinding()]
param(
    [string]$DahuaWheelPath = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (!(Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Install Python 3.12 x64 first."
}

$Version = & py -3.12 --version 2>&1

if ($LASTEXITCODE -ne 0 -or "$Version" -notmatch "Python 3\.12\.") {
    throw "Python 3.12 is required."
}

if (!(Test-Path ".\.venv\Scripts\python.exe")) {
    & py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Venv creation failed."
    }
}

$Python = ".\.venv\Scripts\python.exe"
$Lock = "configs\dependencies\windows-py312-cu130.lock.txt"

Write-Host "INSTALLING LOCKED DEPENDENCIES"

& $Python -m pip install -r $Lock
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed."
}

Write-Host "INSTALLING EDITABLE PROJECT"

& $Python -m pip install -e ".[dev]"
if ($LASTEXITCODE -ne 0) {
    throw "Editable installation failed."
}

if ($DahuaWheelPath) {
    if (!(Test-Path $DahuaWheelPath)) {
        throw "Dahua Wheel not found."
    }

    & $Python -m pip install $DahuaWheelPath
    if ($LASTEXITCODE -ne 0) {
        throw "Dahua SDK installation failed."
    }
}
else {
    Write-Warning "Dahua SDK must be installed separately from its official distribution."
}

Write-Host "CHECKING DEPENDENCIES"
& $Python -m pip check
if ($LASTEXITCODE -ne 0) {
    throw "pip check failed."
}

Write-Host "CHECKING CUDA"
& $Python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

if ($LASTEXITCODE -ne 0) {
    throw "PyTorch check failed."
}

Write-Host "RUNNING TESTS"
& $Python -m pytest -q
if ($LASTEXITCODE -ne 0) {
    throw "Tests failed."
}

Write-Host "SETUP COMPLETE"
