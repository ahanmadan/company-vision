# ARCHITECTURE — Company Vision v1.0

- وضعیت: پیشنهادی برای تصویب در گره 1.0
- اصل: هسته مشاهده مستقل از موتور استنتاج رفتار

## ۱. معماری منطقی
```text
Camera(s) / Video File
    -> Capture Adapters (OpenCV)
    -> FramePacket(timestamp, camera_id, sequence, dimensions)
    -> VisionPipeline
         +-> YOLO Pose -> Person boxes + COCO keypoints -> ByteTrack IDs
         +-> YOLO Detection -> Object boxes/classes/confidence
    -> FrameResult (canonical, one result per processed frame)
    -> +-> Overlay Renderer / Metrics
       +-> BehaviorPlugin Registry -> TemporalState -> BehaviorEvents
       +-> Event Storage -> FastAPI -> Authorized Dashboard
```

## ۲. تصمیم‌های مرزی
1. **دوربین:** اتصال، شمارنده فریم، reconnect و محدودسازی بافر؛ هیچ آگاهی از YOLO و رفتار ندارد.
2. **Vision:** مدل Pose منبع person box و track ID است؛ Detector دیگر مسئول اشیا است. دو خروجی از *همان فریم* استخراج می‌شوند.
3. **Tracking:** ByteTrack به هر camera_id و session_id مستقل وابسته است؛ Track ID هویت کارمند نیست و پس از restart قابل بازاستفاده است.
4. **Observation:** FrameResult حاوی مشاهده خام، confidence، مختصات و نبود داده است؛ وابسته به نمایشگر یا دیتابیس نیست.
5. **Behavior:** فقط FrameResult و تاریخچه مجاز را دریافت می‌کند؛ اجازه دستکاری دوربین/مدل را ندارد.
6. **Presentation/Storage:** فقط مصرف‌کننده خروجی‌اند؛ شکست آن‌ها نباید باعث توقف پردازش اصلی شود.

## ۳. قراردادهای داده (نسخه مفهومی؛ اجرای Pydantic در 1.4)
- `FramePacket`: schema_version, camera_id, session_id, frame_id, captured_at_utc, monotonic_timestamp, width, height, pixel_format, frame_data.
- `PersonObservation`: track_id(optional), bbox_xyxy, detector_confidence, keypoints_by_name {x,y,confidence,valid} و status.
- `ObjectObservation`: class_id, class_name, bbox_xyxy, confidence, object_id(optional).
- `FrameResult`: schema_version, camera_id, session_id, frame_id, timestamps, dimensions, persons[], objects[], processing_ms, warnings[].
- `BehaviorEvent`: event_id, camera_id, session_id, track_id(optional), behavior_type, start/end, confidence(optional), status {candidate,confirmed,unknown,ended}, supporting_evidence, model_version, rule_version.
- مختصات پیکسلی بر اساس فریم منبع تعریف می‌شوند. هر تغییر رزولوشن باید همراه تبدیل مختصات ثبت شود.
- Confidence مدل تشخیص *احتمال کالیبره‌شده رفتار* نیست؛ نباید مستقیم به درصد یقین رفتار تبدیل شود.

## ۴. روش اتصال Pose و Detection
- YOLO Pose برای افراد، Bounding Box و نقاط بدن را می‌دهد.
- ByteTrack روی افراد اجرا شده و ID های محلی می‌دهد.
- YOLO Detection فقط جعبه و کلاس اشیا را می‌دهد؛ انتساب گوشی به شخص در هسته انجام نمی‌شود.
- ابزار انتساب اشیا به شخص، در آینده جزئی از Plugin با قواعد فاصله نرمال‌شده و تاریخچه است.

## ۵. چند دوربین و کارایی
- برای هر دوربین session_id و tracker state مجزا؛ هر دوربین صف ورودی محدود و سیاست حذف فریم قدیمی در حالت live دارد.
- اتصال و inference مستقل طراحی می‌شوند؛ فریم‌های stale نباید باعث انباشته‌شدن تأخیر شوند.
- Pipeline ابتدا ترتیبی و ساده پیاده می‌شود، سپس بر اساس Benchmark به concurrency/اشتراک GPU توسعه می‌یابد.
- هر دوربین تنظیمات ROI، رزولوشن، source و سلامت خودش را دارد.
- فرایند reconnect باید session جدید بسازد و state ByteTrack و تاریخچه مرتبط را بی‌اعتبار کند.

## ۶. نمای نیم‌رخ و محدودیت مشاهده
- نقاط بدن قابل مشاهده با confidence معتبر مصرف شوند؛ missing/occluded وضعیت مستقل دارند.
- تشخیص اشیای کوچک با ROI هدفمند *پس از Benchmark* بررسی می‌شود؛ بزرگ‌نمایی مصنوعی جزئیات ناپدیدشده را بازنمی‌گرداند.
- زاویه، نور و میزان پوشیدگی با داده واقعی آزموده می‌شوند؛ تصمیم درباره Fine-tuning یا تغییر زاویه در فاز ۴.
- هیچ قانون رفتاری تنها از نبود گوشی یا نبود حرکت به خواب قطعی نرسد.

## ۷. امنیت، حریم خصوصی و حقوق استفاده
- RTSP credential از محیط امن/secret config گرفته شود؛ در مخزن و لاگ ثبت نشود.
- داده‌های خام و مدل‌های دانلودی در Git نگهداری نشوند؛ نقش‌ها و تاریخچه دسترسی محدود شوند.
- ذخیره ویدئوی خام به صورت پیش‌فرض غیرفعال؛ دوره نگهداری و حذف براساس مقررات و سیاست مصوب تعیین شود.
- بررسی مجوزهای Ultralytics، PyTorch و توزیع محصول پیش از استقرار اجباری است.

## ۸. ساختار پروژه مقصد
```text
company-vision/
  docs/{WORKFLOW.md,PROJECT_SCOPE.md,ARCHITECTURE.md,DECISIONS.md,PROGRESS.md,TEST_REPORTS/}
  configs/{app.yaml,cameras.yaml,models.yaml,behaviors.yaml}
  src/company_vision/
    camera/{base.py,video_source.py,webcam_source.py,rtsp_source.py,frame_buffer.py}
    vision/{pose.py,detection.py,tracking.py,pipeline.py}
    core/{schemas.py,settings.py,events.py}
    behaviors/{base.py,registry.py,temporal.py,plugins/}
    storage/ api/ dashboard/ rendering/ utils/
  tests/{unit/,integration/,performance/,fixtures/}
  scripts/ data/{samples/,outputs/} models/
  pyproject.toml .env.example .gitignore README.md
```

## ۹. سیاست تغییر معماری
هر تغییر قرارداد FrameResult یا رابطه پلاگین با هسته، یک ADR جدید در DECISIONS.md، به‌روزرسانی schema_version، تست رگرسیون و ثبت مهاجرت سازگاری می‌خواهد. افزودن behavior plugin نباید نیازمند تغییر CameraSource، PoseDetector، ObjectDetector یا Tracker باشد.

## 10. Phase 5 — Independent Behavior Recognition

The behavior engine consumes validated observation contracts.
It does not control camera capture, YOLO inference or tracking.

The approved seven-node structure is:

1. BehaviorInputAdapter
2. FeatureExtractor
3. TemporalStateManager
4. BehaviorPlugin Registry
5. Rule Engine
6. Event Manager
7. BehaviorResult

The initial behavior plugins are PhoneUsagePlugin,
InactivityPlugin and GatheringPlugin.

Node 5.1 introduces an independent BehaviorFrame contract.
The existing FrameResult contract is not modified.

Session identity and monotonic time must be supplied by the
upstream orchestration layer. The adapter must not fabricate them.

A person becomes behavior-eligible only when explicit observed
and lifecycle evidence supports that eligibility.

See `docs/phase5/PHASE5_ARCHITECTURE.md`.
