
# Company Vision v1.0
# Phase 1 - Final Verification
# Read-only checks, except writing the audit report.

$ErrorActionPreference = "Stop"

$Root = "D:\Projects\company-vision"
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Report = Join-Path $Root "docs\TEST_REPORTS\PHASE_1_FINAL_VERIFICATION.md"

Set-Location $Root

$Results = @()
$Failures = 0

function Invoke-Audit {
    param(
        [string]$Name,
        [scriptblock]$Action
    )

    Write-Host ""
    Write-Host "======================================"
    Write-Host "CHECK: $Name"
    Write-Host "======================================"

    $global:LASTEXITCODE = 0

    try {
        & $Action

        if ($LASTEXITCODE -ne 0) {
            throw "Command returned exit code $LASTEXITCODE"
        }

        $script:Results += [PSCustomObject]@{
            Name = $Name
            Status = "PASS"
            Details = "Verification succeeded"
        }

        Write-Host "RESULT: PASS" -ForegroundColor Green
    }
    catch {
        $script:Failures++

        $script:Results += [PSCustomObject]@{
            Name = $Name
            Status = "FAIL"
            Details = $_.Exception.Message
        }

        Write-Host "RESULT: FAIL" -ForegroundColor Red
        Write-Host $_.Exception.Message
    }
}

# ------------------------------------
# 01 - PROJECT STRUCTURE
# ------------------------------------

Invoke-Audit "01 - Project Structure" {

    $Required = @(
        ".git",
        ".gitignore",
        ".env.example",
        "README.md",
        "pyproject.toml",
        "docs\PROJECT_SCOPE.md",
        "docs\ARCHITECTURE.md",
        "docs\WORKFLOW.md",
        "docs\DECISIONS.md",
        "docs\PROGRESS.md",
        "configs\app.yaml",
        "src\company_vision\__init__.py",
        "src\company_vision\core\settings.py",
        "src\company_vision\core\schemas.py",
        "src\company_vision\utils\logging_config.py",
        "scripts\check_node_1_4.py",
        "tests\unit\test_node_1_4.py",
        "configs\dependencies\windows-py312-cu130.lock.txt"
    )

    foreach ($Item in $Required) {
        if (-not (Test-Path $Item)) {
            throw "Missing required path: $Item"
        }
    }

    Write-Host "Required paths: $($Required.Count)"
    Write-Host "Structure: VALID"
}

# ------------------------------------
# 02 - ALL PYTEST TESTS
# ------------------------------------

Invoke-Audit "02 - Pytest" {

    & $Python -m pytest tests -q

    if ($LASTEXITCODE -ne 0) {
        throw "Pytest failed"
    }
}

# ------------------------------------
# 03 - RUFF CODE QUALITY
# ------------------------------------

Invoke-Audit "03 - Ruff" {

    & $Python -m ruff check src tests scripts

    if ($LASTEXITCODE -ne 0) {
        throw "Ruff checks failed"
    }
}

# ------------------------------------
# 04 - YAML AND SETTINGS
# ------------------------------------

Invoke-Audit "04 - YAML Settings" {

    @'
from company_vision.core.settings import load_settings

settings = load_settings("configs/app.yaml")

assert settings.app.name == "Company Vision"
assert settings.camera.camera_id == "office_cam_01"
assert settings.camera.target_fps > 0
assert 0 <= settings.models.pose_confidence <= 1
assert 0 <= settings.models.detection_confidence <= 1
assert settings.privacy.save_raw_frames is False

print("YAML SETTINGS: PASS")
'@ | & $Python -

    if ($LASTEXITCODE -ne 0) {
        throw "YAML validation failed"
    }
}

# ------------------------------------
# 05 - DATA CONTRACTS
# ------------------------------------

Invoke-Audit "05 - Frame Data Contracts" {

    @'
from datetime import datetime

import numpy as np
from pydantic import ValidationError

from company_vision.core.schemas import (
    BoundingBox,
    FramePacket,
    FrameResult,
    Keypoint,
    PersonObservation,
    ObjectObservation,
    current_utc,
)

image = np.zeros((120, 160, 3), dtype=np.uint8)

packet = FramePacket(
    camera_id="test_cam",
    frame_id=100,
    timestamp=current_utc(),
    image=image,
)

person = PersonObservation(
    track_id=1,
    bbox=BoundingBox(
        x1=10, y1=10, x2=100, y2=110
    ),
    confidence=0.95,
    keypoints=[
        Keypoint(
            name="left_wrist",
            x=50, y=60, confidence=0.90
        )
    ],
)

phone = ObjectObservation(
    class_id=67,
    class_name="cell phone",
    bbox=BoundingBox(
        x1=45, y1=55, x2=65, y2=85
    ),
    confidence=0.88,
)

result = FrameResult.from_packet(
    packet=packet,
    persons=[person],
    objects=[phone],
)

assert result.frame_id == 100
assert result.width == 160
assert result.height == 120
assert len(result.persons) == 1
assert len(result.objects) == 1

restored = FrameResult.model_validate_json(
    result.model_dump_json()
)

assert restored == result

# Negative test: invalid bounding box
try:
    BoundingBox(x1=100, y1=10, x2=20, y2=90)
except ValidationError:
    pass
else:
    raise AssertionError("Invalid bounding box accepted")

# Negative test: duplicate tracking ID
try:
    FrameResult.from_packet(
        packet=packet,
        persons=[person, person],
    )
except ValidationError:
    pass
else:
    raise AssertionError("Duplicate track ID accepted")

# Negative test: naive timestamp
try:
    FramePacket(
        camera_id="test_cam",
        frame_id=1,
        timestamp=datetime(2026, 1, 1),
        image=image,
    )
except ValueError:
    pass
else:
    raise AssertionError("Naive timestamp accepted")

print("FRAME CONTRACTS: PASS")
'@ | & $Python -

    if ($LASTEXITCODE -ne 0) {
        throw "Frame contracts failed"
    }
}

# ------------------------------------
# 06 - LOGGING AND SMOKE TEST
# ------------------------------------

Invoke-Audit "06 - Logging and Smoke Test" {

    & $Python scripts/check_node_1_4.py

    if ($LASTEXITCODE -ne 0) {
        throw "Smoke test failed"
    }
}

# ------------------------------------
# 07 - DEPENDENCIES AND CUDA
# ------------------------------------

Invoke-Audit "07 - Dependencies and CUDA" {

    & $Python -m pip check

    if ($LASTEXITCODE -ne 0) {
        throw "Dependency conflicts found"
    }

    @'
import sys
import torch
import torchvision
import cv2
import numpy
import yaml
import pydantic
import ultralytics
import pytest
import company_vision

assert sys.version_info[:2] == (3, 12)
assert sys.prefix != sys.base_prefix

assert torch.cuda.is_available()
assert torch.version.cuda is not None

gpu = torch.cuda.get_device_name(0)
assert "4070 SUPER" in gpu.upper()

x = torch.ones((128, 128), device="cuda")
y = x @ x
torch.cuda.synchronize()

assert bool(torch.all(y == 128).item())

print("PYTHON:", sys.version.split()[0])
print("TORCH:", torch.__version__)
print("TORCHVISION:", torchvision.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", gpu)
print("OPENCV:", cv2.__version__)
print("NUMPY:", numpy.__version__)
print("ULTRALYTICS:", ultralytics.__version__)
print("CUDA COMPUTE: PASS")
'@ | & $Python -

    if ($LASTEXITCODE -ne 0) {
        throw "CUDA or import verification failed"
    }
}

# ------------------------------------
# 08 - GIT REPOSITORY
# ------------------------------------

Invoke-Audit "08 - Git Repository" {

    $Repository = (git rev-parse --show-toplevel).Trim()

    if ($LASTEXITCODE -ne 0) {
        throw "Git repository not available"
    }

    if ($Repository.Replace("\", "/") -ne $Root.Replace("\", "/")) {
        throw "Incorrect Git repository root"
    }

    $Branch = (git branch --show-current).Trim()

    if ($Branch -ne "main") {
        throw "Expected main branch, got: $Branch"
    }

    git log -1 --oneline

    if ($LASTEXITCODE -ne 0) {
        throw "No valid Git commit found"
    }

    Write-Host "Git branch: $Branch"
    Write-Host "Git working tree:"
    git status --short

    if ($LASTEXITCODE -ne 0) {
        throw "Git status failed"
    }
}

# ------------------------------------
# FINAL REPORT
# ------------------------------------

New-Item -ItemType Directory -Force (
    Split-Path $Report -Parent
) | Out-Null

$Passed = @($Results | Where-Object {
    $_.Status -eq "PASS"
}).Count

$Lines = @(
    "# Company Vision - Phase 1 Final Verification",
    "",
    "- Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
    "- Phase: 1",
    "- Nodes: 1.0 to 1.4",
    "- Passed: $Passed / 8",
    "- Failed: $Failures",
    "",
    "## Results",
    "",
    "| Check | Status | Details |",
    "|---|---|---|"
)

foreach ($Entry in $Results) {
    $SafeDetails = $Entry.Details.Replace("|", "/")
    $Lines += "| $($Entry.Name) | $($Entry.Status) | $SafeDetails |"
}

$Lines += ""
$Lines += "## Final Result"
$Lines += ""

if ($Failures -eq 0) {
    $Lines += "PASS - All eight audit checks succeeded."
} else {
    $Lines += "FAIL - Correct failing checks before Phase 2."
}

$Lines | Set-Content $Report -Encoding UTF8

Write-Host ""
Write-Host "======================================"
Write-Host "COMPANY VISION - PHASE 1 FINAL AUDIT"
Write-Host "======================================"
Write-Host "PASSED: $Passed / 8"
Write-Host "FAILED: $Failures"
Write-Host "REPORT: $Report"

if ($Failures -eq 0) {
    Write-Host "PHASE 1 AUDIT: PASS" -ForegroundColor Green
    exit 0
}

Write-Host "PHASE 1 AUDIT: FAIL" -ForegroundColor Red
exit 1
