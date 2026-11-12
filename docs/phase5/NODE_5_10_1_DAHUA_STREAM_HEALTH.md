# Node 5.10.1 - Dahua NetSDK Stream Health

## Objective
Validate a real Dahua stream using NetSDK.

## Acceptance criteria
1. NetSDK DLL loads.
2. SDK initialization succeeds.
3. Device login succeeds.
4. RealPlay handle is obtained.
5. A new native stream sample is recorded.
6. The sample exceeds 32768 bytes.
7. FFmpeg decodes an actual frame.
8. OpenCV confirms valid frame dimensions.
9. Unit tests and full regression pass.

## Required configuration
CV_NETSDK_DLL
CV_DAHUA_HOST
CV_DAHUA_USER
CV_DAHUA_PASSWORD

Optional:
CV_DAHUA_PORT

## Security
Credentials are not stored in repository files.
Camera samples remain local.
Do not commit local validation artifacts.

## Compatibility
The native DLL must match the Python process
architecture and the installed Dahua SDK.
Native API signatures must match the SDK build.

## Exclusions
No YOLO inference.
No GPU benchmarking.
No behavior accuracy claims.
No Track ID evaluation.
No video overlay rendering.

## Next
Node 5.10.2: GPU inference and integration validation.