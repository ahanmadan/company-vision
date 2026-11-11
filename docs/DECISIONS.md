# DECISIONS — Company Vision v1.0

- سبک ثبت: ADR (Architecture Decision Record)
- وضعیت این سند: پیش‌نویس منتظر تأیید در گره 1.0
- هر تصمیم جدید: ID یکتا، وضعیت، زمینه، انتخاب، دلیل، پیامد، تاریخ

## ADR-001 — معماری مبتنی بر لایه و پلاگین
- وضعیت: PROPOSED
- تاریخ: 2026-10-28
- زمینه: قابلیت‌های رفتاری بعدی نباید موتور تصویر را به هم بزنند.
- تصمیم: Capture، Vision، Observation، Behavior، Storage/API/UI مستقل هستند؛ plugin فقط قرارداد FrameResult را می‌خواند.
- پیامد: تست و نسخه‌بندی قرارداد لازم است.

## ADR-002 — منبع واحد مشاهده فرد
- وضعیت: PROPOSED
- تصمیم: YOLO Pose مسئول bbox/keypoints فرد است؛ ByteTrack مسئول شناسه موقت؛ YOLO Detection فقط اشیا را شناسایی می‌کند.
- دلیل: جلوگیری از دو تشخیص موازی شخص و خطای پیوند.
- پیامد: محدودیت‌های Pose و پوشیدگی باید در فاز ۴ سنجیده شوند.

## ADR-003 — مدیریت Track و Session
- وضعیت: PROPOSED
- تصمیم: شناسه فرد محلی به camera_id و session_id است؛ reconnect، Session و state جدید ایجاد می‌کند.
- پیامد: track_id به‌تنهایی شناسه جهانی یا هویت کارمند نیست.

## ADR-004 — پردازش زنده و کنترل تأخیر
- وضعیت: PROPOSED
- تصمیم: بافر bounded برای ویدئوی live؛ امکان حذف فریم‌های قدیمی و ثبت dropped_frames؛ اولویت تازگی بر تحلیل همه فریم‌های ورودی.
- پیامد: FPS پردازش‌شده از FPS دوربین تفکیک می‌شود.

## ADR-005 — برچسب‌گذاری رفتار همراه با عدم قطعیت
- وضعیت: PROPOSED
- تصمیم: هیچ نتیجه «خواب قطعی» صرفاً از بی‌حرکتی یا ندیدن گوشی صادر نمی‌شود؛ unknown باید قابل نمایش و تست باشد.
- پیامد: وقایع رفتاری نیازمند زمان و ارزیابی مستقل‌اند.

## ADR-006 — پلتفرم و ابزارهای اولیه
- وضعیت: PROPOSED
- تصمیم: Windows + VS Code + Python 3.12؛ OpenCV، YOLO Pose/Detection، ByteTrack؛ FastAPI/PostgreSQL در فاز ۶.
- پیامد: نسخه دقیق بسته‌ها، CPU/GPU و مجوزها بعداً تثبیت می‌شوند.

## ADR-007 — حریم خصوصی و نگهداری داده
- وضعیت: PROPOSED
- تصمیم: پردازش محلی ترجیحی؛ ضبط دائمی ویدئو پیش‌فرض غیرفعال؛ اصول دسترسی حداقلی و سیاست حذف پیش از استقرار.
- پیامد: نیازمند تأیید حقوقی/سازمانی متناسب با محل استقرار است.

## موارد باز برای گره‌های بعدی (نه مانع تصویب چارچوب)
1. مشخصات دستگاه ضبط/RTSP، رزولوشن، FPS و زاویه واقعی دفتر (1.1 و 2.3).
2. CPU، GPU، RAM و بودجه محاسباتی (1.1).
3. وزن و نسخه دقیق YOLO و وابستگی‌های تثبیت‌شده پس از بررسی مجوز (1.3 و 3.1).
4. معیارهای کمی Precision/Recall و حد تأخیر بر مبنای Benchmark (3.6، 5.6، 8.3).
5. دوره نگهداری و مبنای قانونی پردازش داده کارکنان (6.4 و قبل از استقرار).
6. نیاز به تشخیص هویت شخص؛ خارج از نسخه 1.0 مگر با تصمیم مستقل.

## قالب ADR آتی
```text
ADR-XXX — Title
Date:
Status: PROPOSED | ACCEPTED | SUPERSEDED
Context:
Decision:
Rationale:
Consequences:
Related tests / migrations:
```

## ADR-008 — Independent Behavior Input Contract

- Date: 2026-11-11
- Status: ACCEPTED
- Context: The behavior engine must not depend on YOLO, camera SDK or tracker internals.
- Decision: Introduce BehaviorInputAdapter and a versioned BehaviorFrame contract.
- Input sources: canonical FrameResult and neutral tracking-output records.
- Identity: preserve camera_id, session_id and track_id; never synthesize identity.
- Time: preserve UTC timestamp and accept monotonic timestamp only when supplied.
- Eligibility: predicted or unconfirmed observations are not automatically eligible.
- Compatibility: do not modify phase 1-4 schemas or tracking algorithms.
- Consequences: future FeatureExtractor and temporal components consume BehaviorFrame only.
- Related implementation: `src/company_vision/behaviors/`.
- Related tests: `tests/unit/test_behavior_input_adapter.py`.

## ADR-009 — Seven-Node Independent Behavior Engine

- Date: 2026-11-11
- Status: ACCEPTED
- Context: The behavior engine must support independent development and future extension without changing the vision pipeline.
- Decision: Phase 5 consists of seven independent components: BehaviorInputAdapter, FeatureExtractor, TemporalStateManager, BehaviorPlugin Registry, Rule Engine, Event Manager and BehaviorResult.
- Plugin policy: PhoneUsagePlugin, InactivityPlugin and GatheringPlugin are plugins managed by the registry, not additional architectural nodes.
- Input boundary: BehaviorInputAdapter converts validated observations into BehaviorFrame. Existing FrameResult remains unchanged.
- Temporal policy: Track history is scoped by camera_id, session_id and track_id.
- Scalability: No fixed person-count limit is introduced.
- Compatibility: Phases 1-4 remain unchanged. SYNC remains the reference pipeline.
- Consequences: Each node has an independent contract, tests and acceptance criteria. The next node cannot begin before the previous node passes.
- Related implementation: `src/company_vision/behaviors/`.
- Related architecture: `docs/phase5/PHASE5_ARCHITECTURE.md`.
## ADR-010 — Stateless Geometric Feature Extraction

- Date: 2026-11-11
- Status: ACCEPTED
- Context: The behavior engine needs a reusable geometry layer independent of tracking and behavior rules.
- Decision: Node 5.2 converts BehaviorFrame into immutable FrameFeatures.
- Weight policy: Explicit keypoint weights are authoritative when supplied; an entirely absent weight map falls back to confidence.
- Geometry: Coordinates are normalized without clipping or modifying source observations.
- Boundaries: Temporal history, classification, rules and events remain outside this node.
- Compatibility: Existing schemas and phases 1-4 remain unchanged.
- Related implementation: src/company_vision/behaviors/feature_extractor.py
- Related tests: tests/unit/test_feature_extractor.py

## ADR-010 — Stateless Geometric Feature Extraction

- Date: 2026-11-11
- Status: ACCEPTED
- Context: The behavior engine needs a reusable geometry layer independent of tracking and behavior rules.
- Decision: Node 5.2 converts BehaviorFrame into immutable FrameFeatures.
- Weight policy: Explicit keypoint weights are authoritative when supplied; an entirely absent weight map falls back to confidence.
- Geometry: Coordinates are normalized without clipping or modifying source observations.
- Boundaries: Temporal history, classification, rules and events remain outside this node.
- Compatibility: Existing schemas and phases 1-4 remain unchanged.
- Related implementation: src/company_vision/behaviors/feature_extractor.py
- Related tests: tests/unit/test_feature_extractor.py

## ADR-010 — Stateless Geometric Feature Extraction

- Date: 2026-11-11
- Status: ACCEPTED
- Context: The behavior engine needs a reusable geometry layer independent of tracking and behavior rules.
- Decision: Node 5.2 converts BehaviorFrame into immutable FrameFeatures.
- Weight policy: Explicit keypoint weights are authoritative when supplied; an entirely absent weight map falls back to confidence.
- Geometry: Coordinates are normalized without clipping or modifying source observations.
- Boundaries: Temporal history, classification, rules and events remain outside this node.
- Compatibility: Existing schemas and phases 1-4 remain unchanged.
- Related implementation: src/company_vision/behaviors/feature_extractor.py
- Related tests: tests/unit/test_feature_extractor.py

## ADR-011 — Independent Behavior Plugin Registry

- Date: 2026-11-11
- Status: ACCEPTED

Context:
The behavior engine requires independently manageable plugins.

Decision:
Node 5.4 introduces a structural BehaviorPlugin protocol
and a thread-safe BehaviorPluginRegistry.

Contract:
- plugin_id
- version
- execute(context)
- BehaviorPluginContext
- PluginResult

Registry capabilities:
- Individual and atomic batch registration
- Duplicate ID rejection
- Plugin lookup and removal
- Enable/disable lifecycle
- Individual and deterministic batch execution
- Result identity validation
- Explicit plugin execution errors

Initial plugin identifiers:
- phone_usage
- inactivity
- gathering

Boundaries:
- No direct YOLO dependency.
- No direct camera SDK dependency.
- No direct tracker dependency.
- No behavior thresholds in the registry.
- No event lifecycle in this node.

Rule evaluation remains in Node 5.5.

Related implementation:
src/company_vision/behaviors/plugin_registry.py

Related tests:
tests/unit/test_behavior_plugin_registry.py

## ADR-010 — Rule Engine Evidence Semantics

- Status: ACCEPTANCE PENDING
- Date: 2026-11-11
- Context: Plugin evidence may be incomplete or unavailable.
- Decision:
  1. Missing evidence must never be converted to negative evidence.
  2. Unknown and not_applicable remain separate outcomes.
  3. Person-level evaluation requires an explicitly eligible, observed, non-predicted track.
  4. Numeric thresholds are supplied by immutable RuleDefinition objects.
  5. Rule evaluation validates camera/session/frame identity.
  6. Rule Engine consumes PluginResult and does not execute plugins.
  7. Event lifecycle remains the responsibility of Node 5.6.
- Consequence: Plugin output schemas and rule configuration must remain explicit and testable.


## ADR-012 — Deterministic Behavior Event Lifecycle

- Date: 2026-11-11
- Status: PROPOSED pending Node 5.6 acceptance

Context:
Rule Engine produces per-frame RuleEvaluation records.
A separate component is required to manage persistent event state.

Decision:
Node 5.6 introduces EventManager.

Event identity:
camera_id + session_id + rule_id + track_id + start_frame_id.

Lifecycle:
started, updated, confirmed and ended.

Policy:
Confirmation and termination frame counts are explicit EventPolicy
configuration values. No behavior thresholds are introduced.

Evidence:
unknown and not_applicable are not negative evidence.
They cannot independently terminate an active event.

Boundaries:
No database persistence, plugin execution, rule evaluation,
camera processing or final BehaviorResult formatting.

Compatibility:
Nodes 5.1-5.5 and Phases 1-4 remain unchanged.

Implementation:
src/company_vision/behaviors/event_manager.py

Tests:
tests/unit/test_event_manager.py

## Node 5.7 — Output Contract Decisions

1. BehaviorResult is independent of database and UI.
2. The output contract is versioned.
3. Event lifecycle history must be preserved.
4. Unknown status must not imply event termination.
5. Rule evaluations and evidence must remain available.
6. Serialization must reject unsupported component types.
7. Real-camera accuracy is not established by unit tests.