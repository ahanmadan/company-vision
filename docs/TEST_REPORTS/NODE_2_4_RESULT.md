- Phase: 2
- Node: 2.4
- Title: Frame Buffer and Reconnection
- Branch: feature/phase2-node-2-4
- Status: IN_PROGRESS

## Implemented Features

- Background frame producer
- FrameSource protocol
- Bounded FIFO queue
- Drop-oldest policy
- Continuous output frame IDs
- Automatic source recreation
- Exponential reconnect backoff
- Connection and frame statistics
- Safe credential-free error type reporting
- Shutdown request and thread join
- Simulated disconnection smoke test

## Test Results

| Test | Expected | Actual |
|---|---|---|
| pip check | PASS | PENDING |
| Node 2.4 unit tests | 16 passed | PENDING |
| Full regression | 78 passed | PENDING |
| Ruff | PASS | PENDING |
| Node 2.1 regression | PASS | PENDING |
| RTSP configuration | PASS | PENDING |
| Reconnect simulation | PASS | PENDING |
| Dependency lock | Verified | PENDING |
| Real RTSP disconnect/reconnect | Deferred | NOT TESTED |
| GitHub synchronization | PASS | PENDING |

## Architecture Decisions

- Original source classes remain unchanged
- Buffer has limited capacity
- Oldest frame is dropped on overflow
- Producer runs in its own thread
- Frame IDs remain continuous across reconnects
- No raw frame storage
- Source errors are not logged with credentials

## Known Limitations

- No company RTSP connection available yet
- Real network recovery not tested
- OpenCV calls may block until backend timeout
- Internal FFmpeg buffering is not controlled here
- Thread shutdown can fail if capture is uninterruptible
- No persistence of buffer statistics
- No long-term monitoring or alerting
- No YOLO inference yet

## Deferred Validation

- Real RTSP stream reception
- Camera power/network interruption
- Recovery after connection restoration
- Real-world stream latency
- H.264/H.265 decoder compatibility

## Approval

- Last Approved Node: 2.3
- Current Node: 2.4
- Approval: PENDING
- Next Node: 2.5