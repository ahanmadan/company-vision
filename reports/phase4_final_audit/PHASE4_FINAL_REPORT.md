# COMPANY VISION
# PHASE 4 FINAL AUDIT

Audit time: 20261111_123621

Overall evidence status: INCOMPLETE

Nodes reviewed: 4.1 through 4.6

Missing evidence nodes: 4.3

This report indexes actual files found in the repository.
It does not invent missing test results or certify
tests that were not executed.

Original evidence files are preserved.


## Node 4.1

Evidence files found: 1

- `reports/phase4_node41/profile_20261110_125245.json` | 1411 bytes | SHA256 `f40723ba92c1d7fe9efef71e3c21c728bbce78f9a0069fc1a91c0ded864f3053`

## Node 4.2

Evidence files found: 3

- `reports/phase4_node42/capture_analysis_20261110_125639.json` | 7446 bytes | SHA256 `5041a84b9c40337a194316eedb1c2a3c2607386aee98b94f14953ec9503582cf`
- `reports/phase4_node42/capture_source_review.txt` | 9700 bytes | SHA256 `5a3af1fbb52a30f705c3c26ee3091751136df56a61ccab2c0e81acd17aab30e4`

### capture_source_review.txt

===== LINES 310-420 =====
 310:
 311:     username = input("Camera Username: ").strip()
 312:     password = getpass("Camera Password: ")
 313:
 314:     sdk = NetClient()
 315:
 316:     initialized = False
 317:     login_id = 0
 318:     play_id = 0
 319:     free_port = None
 320:
 321:     stream_open = False
 322:     player_started = False
 323:
 324:     lock = Lock()
 325:
 326:     state = {
 327:         "latest": None,
 328:         "sequence": 0,
 329:         "packets": 0,
 330:         "decoded": 0,
 331:         "errors": 0,
 332:     }
 333:
 334:     # ========================================================
 335:     # DECODE CALLBACK
 336:     # ========================================================
 337:
 338:     def on_decoded(
 339:         port,
 340:         buffer,
 341:         buffer_size,
 342:         frame_info,
 343:         user_data,
 344:         reserved,
 345:     ):
 346:         try:
 347:             info = frame_info.contents
 348:
 349:             if info.nType != 3:
 350:                 return
 351:
 352:             width = int(info.nWidth)
 353:             height = int(info.nHeight)
 354:
 355:             if width <= 0 or height <= 0:
 356:                 return
 357:
 358:             if width % 2 or height % 2:
 359:                 return
 360:
 361:             expected = width * height * 3 // 2
 362:
 363:             if buffer_size < expected:
 364:                 return
 365:
 366:             raw = string_at(buffer, expected)
 367:
 368:             with lock:
 369:                 state["sequence"] += 1
 370:                 state["decoded"] += 1
 371:
 372:                 state["latest"] = (
 373:                     state["sequence"],
 374:                     raw,
 375:                     width,
 376:                     height,
 377:                 )
 378:
 379:         except Exception:  # noqa: BLE001
 380:             with lock:
 381:                 state["errors"] += 1
 382:
 383:     # ========================================================
 384:     # REAL DATA CALLBACK
 385:     # ========================================================
 386:
 387:     def on_real_data(
 388:         real_handle,
 389:         data_type,
 390:         buffer,
 391:         buffer_size,
 392:         param,
 393:         user_data,
 394:     ):
 395:         if buffer_size <= 0:
 396:             return
 397:
 398:         try:
 399:             accepted = sdk.InputData(
 400:                 free_port,
 401:                 buffer,
 402:                 buffer_size,
 403:             )
 404:
 405:             with lock:
 406:                 if accepted:
 407:                     state["packets"] += 1
 408:                 else:
 409:                     state["errors"] += 1
 410:
 411:         except Exception:  # noqa: BLE001
 412:             with lock:
 413:                 state["errors"] += 1
 414:
 415:     decoding_callback = fDecCBFun(on_decoded)
 416:     data_callback = fRealDataCallBackEx2(on_real_data)
 417:
 418:     window_name = "Company Vision - Dahua Live YOLO Pose"
 419:
 420:     last_sequence = 0

===== LINES 510-630 =====
 510:
 511:         print("REALPLAY: PASS")
 512:         print("YOLO POSE: READY")
 513:         print("ACTIVE KEYPOINTS: HEAD + SHOULDERS + ELBOWS")
 514:         print("ACTIVE KEYPOINT COUNT: 9")
 515:         print("Waiting for camera frame...")
 516:
 517:         startup_deadline = time.monotonic() + 15
 518:
 519:         while time.monotonic() < startup_deadline:
 520:
 521:             with lock:
 522:                 available = state["latest"] is not None
 523:
 524:             if available:
 525:                 break
 526:
 527:             time.sleep(0.05)
 528:
 529:         if not available:
 530:             raise RuntimeError("No camera frame received")
 531:
 532:         cv2.namedWindow(
 533:             window_name,
 534:             cv2.WINDOW_NORMAL,
 535:         )
 536:
 537:         cv2.resizeWindow(
 538:             window_name,
 539:             1280,
 540:             720,
 541:         )
 542:
 543:         started_at = time.monotonic()
 544:
 545:         track_lifecycle_guard = TrackLifecycleGuard(
 546:             max_hidden_seconds=0.75,
 547:             min_confirmed_observations=2,
 548:         )
 549:
 550:         pose_manager = StablePoseManager(
 551:             confirm_hits=args.confirm_frames,
 552:             max_missed=args.max_missed,
 553:             min_keypoints=3,
 554:             keypoint_conf=args.keypoint_conf,
 555:             ema_alpha=args.ema_alpha,
 556:             match_distance=0.85,
 557:             pose_filter_mode="one_euro",
 558:             new_track_min_conf=0.50,
 559: )
 560:
 561:         print("\nLIVE YOLO POSE STARTED")
 562:         print("Press Q to stop.")
 563:
 564:         # ====================================================
 565:         # LIVE PROCESSING LOOP
 566:         # ====================================================
 567:
 568:         while True:
 569:
 570:             with lock:
 571:                 item = state["latest"]
 572:
 573:             if item is None:
 574:
 575:                 if cv2.waitKey(1) & 0xFF == ord("q"):
 576:                     break
 577:
 578:                 continue
 579:
 580:             sequence, raw, width, height = item
 581:
 582:             if sequence == last_sequence:
 583:
 584:                 if cv2.waitKey(1) & 0xFF == ord("q"):
 585:                     break
 586:
 587:                 time.sleep(0.005)
 588:                 continue
 589:
 590:             last_sequence = sequence
 591:
 592:             yuv = np.frombuffer(
 593:                 raw,
 594:                 dtype=np.uint8,
 595:             ).reshape((height * 3 // 2, width))
 596:
 597:             frame = cv2.cvtColor(
 598:                 yuv,
 599:                 cv2.COLOR_YUV2BGR_I420,
 600:             )
 601:
 602:             # =================================================
 603:             # YOLO26X POSE - FULL FRAME
 604:             # =================================================
 605:
 606:             inference_start = time.perf_counter()
 607:
 608:             results = model.predict(
 609:                 source=frame,
 610:                 device="cuda:0",
 611:                 imgsz=args.imgsz,
 612:                 conf=args.conf,
 613:                 classes=[0],
 614:                 verbose=False,
 615:             )
 616:
 617:             inference_ms = (
 618:                 time.perf_counter() - inference_start
 619:             ) * 1000
 620:
 621:             inference_times.append(inference_ms)
 622:
 623:             result = results[0]
 624:
 625:             boxes = result.boxes
 626:             keypoints_object = result.keypoints
 627:
 628:             if (
 629:                 boxes is not None
 630:                 and keypoints_object is not None

===== LINES 680-715 =====
 680:
 681:                 poses = np.empty(
 682:                     (0, 9, 3),
 683:                     dtype=np.float32,
 684:                 )
 685:
 686:                 scores = np.empty(
 687:                     (0,),
 688:                     dtype=np.float32,
 689:                 )
 690:
 691:             # =================================================
 692:             # TEMPORAL TRACKING
 693:             # =================================================
 694:
 695:             tracked_persons = pose_manager.update(
 696:                 coordinates,
 697:                 poses,
 698:                 scores,
 699:             )
 700:
 701:             tracked_persons = track_lifecycle_guard.update(
 702:                 tracked_persons,
 703:                 timestamp=time.monotonic(),
 704:             )
 705:
 706:             annotated = frame.copy()
 707:
 708:             person_count = sum(
 709:                 1
 710:                 for person in tracked_persons
 711:                 if person["observed"]
 712:             )
 713:
 714:             predicted_count = sum(
 715:                 1

===== LINES 780-815 =====
 780:
 781:             if cv2.waitKey(1) & 0xFF == ord("q"):
 782:                 break
 783:
 784:             now = time.monotonic()
 785:
 786:             # =================================================
 787:             # TERMINAL REPORT
 788:             # =================================================
 789:
 790:             if now - last_report >= 5:
 791:
 792:                 with lock:
 793:                     decoded = state["decoded"]
 794:                     packets = state["packets"]
 795:
 796:                 print(
 797:                     f"FPS={processing_fps:.2f} "
 798:                     f"DECODED={decoded} "
 799:                     f"PROCESSED={processed_frames} "
 800:                     f"PACKETS={packets} "
 801:                     f"PERSONS={person_count} "
 802:                     f"INFERENCE={inference_ms:.1f}ms"
 803:                 )
 804:
 805:                 last_report = now
 806:
 807:     except KeyboardInterrupt:
 808:         print("\nStopped by user")
 809:
 810:     finally:
 811:
 812:         print("\nStopping live processing...")
 813:
 814:         cv2.destroyAllWindows()
 815:


- `reports/phase4_node42/live_test_20261110_131106.log` | 2368 bytes | SHA256 `5ae65ad1c8e63b1e4398a45f68fd23de8dcd1a66e9ea556d02f022707d2bddce`

### live_test_20261110_131106.log

��G P U :   N V I D I A   G e F o r c e   R T X   4 0 7 0   S U P E R 
 
 L o a d i n g   Y O L O   P o s e . . . 
 
 P O S E   M O D E L :   y o l o 2 6 x - p o s e . p t 
 
 C a m e r a   U s e r n a m e :   
 
 I n i t i a l i z i n g   D a h u a   S D K . . . 
 
 L O G I N :   P A S S 
 
 C H A N N E L S :   8 
 
 R E A L P L A Y :   P A S S 
 
 Y O L O   P O S E :   R E A D Y 
 
 A C T I V E   K E Y P O I N T S :   H E A D   +   S H O U L D E R S   +   E L B O W S 
 
 A C T I V E   K E Y P O I N T   C O U N T :   9 
 
 W a i t i n g   f o r   c a m e r a   f r a m e . . . 
 
 
 
 L I V E   Y O L O   P O S E   S T A R T E D 
 
 P r e s s   Q   t o   s t o p . 
 
 F P S = 1 5 . 0 0   D E C O D E D = 1 0 8   P R O C E S S E D = 6 4   P A C K E T S = 1 6 9   P E R S O N S = 1   I N F E R E N C E = 3 3 . 3 m s 
 
 F P S = 1 9 . 0 7   D E C O D E D = 2 3 4   P R O C E S S E D = 1 7 7   P A C K E T S = 3 6 2   P E R S O N S = 1   I N F E R E N C E = 3 4 . 0 m s 
 
 F P S = 2 0 . 2 6   D E C O D E D = 3 5 9   P R O C E S S E D = 2 9 0   P A C K E T S = 5 5 6   P E R S O N S = 1   I N F E R E N C E = 3 2 . 9 m s 
 
 F P S = 2 0 . 8 2   D E C O D E D = 4 8 5   P R O C E S S E D = 4 0 3   P A C K E T S = 7 5 0   P E R S O N S = 1   I N F E R E N C E = 3 2 . 5 m s 
 
 F P S = 2 1 . 1 0   D E C O D E D = 6 1 0   P R O C E S S E D = 5 1 4   P A C K E T S = 9 4 2   P E R S O N S = 1   I N F E R E N C E = 3 3 . 8 m s 
 
 F P S = 2 1 . 3 0   D E C O D E D = 7 3 6   P R O C E S S E D = 6 2 6   P A C K E T S = 1 1 3 6   P E R S O N S = 1   I N F E R E N C E = 3 3 . 3 m s 
 
 F P S = 2 1 . 4 3   D E C O D E D = 8 6 1   P R O C E S S E D = 7 3 7   P A C K E T S = 1 3 2 7   P E R S O N S = 1   I N F E R E N C E = 3 3 . 4 m s 
 
 F P S = 2 1 . 5 4   D E C O D E D = 9 8 7   P R O C E S S E D = 8 4 9   P A C K E T S = 1 5 2 1   P E R S O N S = 1   I N F E R E N C E = 3 3 . 6 m s 
 
 F P S = 2 1 . 5 7   D E C O D E D = 1 1 1 2   P R O C E S S E D = 9 5 8   P A C K E T S = 1 7 1 4   P E R S O N S = 1   I N F E R E N C E = 3 3 . 0 m s 
 
 F P S = 2 1 . 3 3   D E C O D E D = 1 2 3 6   P R O C E S S E D = 1 0 5 4   P A C K E T S = 1 9 0 7   P E R S O N S = 1   I N F E R E N C E = 3 6 . 3 m s 
 
 F P S = 2 1 . 0 5   D E C O D E D = 1 3 6 1   P R O C E S S E D = 1 1 4 5   P A C K E T S = 2 1 0 0   P E R S O N S = 1   I N F E R E N C E = 3 5 . 2 m s 
 
 


## Node 4.3

**EVIDENCE STATUS: MISSING**


## Node 4.4

Evidence files found: 9

- `reports/phase4_node44/copy_20261110_142237.json` | 231 bytes | SHA256 `689e321890b5c35d93ce47e0625259e669bd20b2c5620190078824c94b6db964`
- `reports/phase4_node44/inplace_20261110_142237.json` | 222 bytes | SHA256 `1a93eec548cee5ddc6e29e83f84c50223ba509ddacfb785def6d1379d7b77937`
- `reports/phase4_node44/node44_final_20261110_144923.json` | 1035 bytes | SHA256 `532606e5ef2ea1f9beebfbb08b68952d97773083bc30a5e6e2381c32f84a195c`
- `reports/phase4_node44/reverse_copy_20261110_144923.json` | 214 bytes | SHA256 `6f9b2d31a297c6b151081ba31753d4fcec43c6fd5718f7fb982ddb9c16415acb`
- `reports/phase4_node44/reverse_inplace_20261110_144020.json` | 222 bytes | SHA256 `a5ff52321f7bcd7fcc0b989e791ba17178fbf78e983ed151fa688bb4337b03f0`
- `reports/phase4_node44/reverse_inplace_20261110_144310.json` | 221 bytes | SHA256 `abdea2766c913b4960da9e3c72d982b81de92f5ec4a8dfead3ededeff23bffe2`
- `reports/phase4_node44/reverse_inplace_20261110_144720.json` | 176 bytes | SHA256 `684bc13b5bd767ba129b65d740f022a73cde850488fb0cb5b3df9a566857350a`
- `reports/phase4_node44/reverse_inplace_20261110_144923.json` | 203 bytes | SHA256 `3719b04bc9059d5330ec6f957448bae3383c1427e47915bff76c3176922e339d`
- `reports/phase4_node44/reverse_inplace_20261110_145550.json` | 221 bytes | SHA256 `b241f026c1ce7acd8a74f6ab89f2ebf1b7de1649a2aeffeb5aba9e0d9818fcb2`

## Node 4.5

Evidence files found: 14

- `reports/node45_live_benchmark_20261110_173400.csv` | 568 bytes | SHA256 `8ad59dd80f0b0ee2b4cd644fb799efee235a5a0fa133b062dc4a4a3a81b739ae`
- `reports/node45_live_benchmark_20261110_173641.csv` | 689 bytes | SHA256 `92b44ad43a521479109b993014df35287dde721f300b050f681c47e4eeb66d9d`
- `reports/node45_live_benchmark_20261110_175147.csv` | 562 bytes | SHA256 `c6bd27c10534aa6c8b6c5f2cffb100dd7e21ab186ed09177931898c8d0275429`
- `reports/node45_live_benchmark_20261110_183813.csv` | 817 bytes | SHA256 `59eb328ef55c47ddc308235a3b5512fa349c62f586be6ed9c7c81230d07bb5b0`
- `reports/node45_live_benchmark_20261110_184816.csv` | 684 bytes | SHA256 `f867c79921e1d850867fc765d703a28e4edeb7f78b9558a2d852ffb330f4436f`
- `reports/node45_live_benchmark_20261110_185229.csv` | 564 bytes | SHA256 `92cd09dab303076c36cddfd91854f976a3d50e53f5db8fc5c31e6798e0735547`
- `reports/node45_live_benchmark_20261110_185804.csv` | 561 bytes | SHA256 `abde0ac221762b900fce0ebad94350dc15886877e2a88735bccf6741b86fdb0f`
- `reports/node45_live_benchmark_20261110_190810.csv` | 562 bytes | SHA256 `f957e1bc28580f2bf3524c2105d7e37ff83e1efdd0acdf1393ee4c404ff2f096`
- `reports/node45_live_benchmark_20261110_191104.csv` | 935 bytes | SHA256 `53eee9c6873f28f4d8d5b6d3cc082ab3a1ff9656c0bb27554e6b1288eb9d7910`
- `reports/node45_live_benchmark_20261111_110204.csv` | 567 bytes | SHA256 `cfa7da0c36c65af7eb7087bb75d20f3e4bf3c01b5d913dd65769f86d5ed945c0`
- `reports/node45_live_benchmark_20261111_110958.csv` | 562 bytes | SHA256 `cf91d75130d2056d5b1153be9ab73e005b93ef7b3e610dce9b7d875dfc521b86`
- `reports/node45_live_benchmark_20261111_111248.csv` | 930 bytes | SHA256 `9b0284c6fe802a34398afeae6599cacd8f4f5543512bcd857c95e2dc425daad2`
- `reports/phase4_node45/node45_final_benchmark_report.json` | 631 bytes | SHA256 `39b86cd10c54df545b37a1f87b6c2f946e1a8aead0d867efea6555025e15cc01`
- `reports/phase4_node45/node45_live_benchmark_20261110_191104.csv` | 931 bytes | SHA256 `10ea1e9e8dae9dabee82b21469d5d617b5818bfbdca343b9d555dab989c4f9be`

## Node 4.6

Evidence files found: 7

- `reports/phase4_node46/cleanup_inventory.txt` | 6604 bytes | SHA256 `cb1b5a4194c76010d1e1fc665a353bfa98e21d5265945d161cd216a4105a0166`

### cleanup_inventory.txt

p.zip
reports/node45_live_benchmark_20261110_173400.csv
reports/node45_live_benchmark_20261110_173641.csv
reports/node45_live_benchmark_20261110_175147.csv
reports/node45_live_benchmark_20261110_183813.csv
reports/node45_live_benchmark_20261110_184816.csv
reports/node45_live_benchmark_20261110_185229.csv
reports/node45_live_benchmark_20261110_185804.csv
reports/node45_live_benchmark_20261110_190810.csv
reports/node45_live_benchmark_20261110_191104.csv
reports/node45_live_benchmark_20261111_110204.csv
reports/phase4_node41/profile_20261110_125245.json
reports/phase4_node42/backups/run_dahua_live_pose_20261110_130832.py
reports/phase4_node42/backups/run_dahua_live_pose_20261110_130900.py
reports/phase4_node42/capture_analysis_20261110_125639.json
reports/phase4_node42/capture_source_review.txt
reports/phase4_node43/backups/pre_quantize_fix_20261110_134542.py
reports/phase4_node43/backups/run_dahua_live_pose_20261110_132034.py
reports/phase4_node44/backups/run_dahua_live_pose_20261110_135349.py
reports/phase4_node44/copy_20261110_142237.json
reports/phase4_node44/inplace_20261110_142237.json
reports/phase4_node44/node44_final_20261110_144923.json
reports/phase4_node44/reverse_copy_20261110_144923.json
reports/phase4_node44/reverse_inplace_20261110_144020.json
reports/phase4_node44/reverse_inplace_20261110_144310.json
reports/phase4_node44/reverse_inplace_20261110_144720.json
reports/phase4_node44/reverse_inplace_20261110_144923.json
reports/phase4_node44/reverse_inplace_20261110_145550.json
reports/phase4_node45/backups/pre_node45_20261110_160519.py
reports/phase4_node45/backups/run_dahua_live_pose.node44_20261110_183449.py
reports/phase4_node45/backups/run_dahua_live_pose.node44_20261110_183515.py
reports/phase4_node45/backups/run_dahua_live_pose.pre_restore.py
reports/phase4_node45/backups/run_dahua_live_pose_20261110_150936.py
reports/phase4_node45/backups/run_dahua_live_pose_20261110_162706.py
reports/phase4_node45/backups/stable_pose_manager.node44_20261110_183449.py
reports/phase4_node45/backups/stable_pose_manager.node44_20261110_183515.py
reports/phase4_node45/current_context/git_status.txt
reports/phase4_node45/current_context/reports/phase4_node44/node44_final_20261110_144923.json
reports/phase4_node45/current_context/scripts/run_dahua_live_pose.py
reports/phase4_node45/current_context/src/company_vision/vision/one_euro_pose_filter.py
reports/phase4_node45/current_context/src/company_vision/vision/stable_pose_manager.py
reports/phase4_node45/current_context/src/company_vision/vision/track_lifecycle_guard.py
reports/phase4_node45/current_context/working_changes.patch
reports/pose_diagnostic_20261110_120615/detections.csv
reports/pose_diagnostic_20261110_120615/frames/frame_000001.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000043.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000088.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000131.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000174.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000217.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000261.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000304.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000348.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000392.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000436.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000480.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000524.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000568.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000612.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000656.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000700.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000744.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000786.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000830.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000874.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000917.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000960.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_001004.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_001048.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_001092.jpg
reports/pose_diagnostic_20261110_120615/r.zip
reports/pose_diagnostic_20261110_120615/validation_evaluation/summary.json
reports/pose_diagnostic_20261110_120615/validation_evaluation/validation_results.csv
reports/step5_backup_20261110_121746/scripts/run_dahua_live_pose.py
reports/step5_backup_20261110_121746/src/company_vision/vision/stable_pose_manager.py
reports/step5_backup_20261110_121746/src/company_vision/vision/track_lifecycle_guard.py
reports/step5_backup_20261110_121928/scripts/run_dahua_live_pose.py
reports/step5_backup_20261110_121928/src/company_vision/vision/stable_pose_manager.py
reports/step5_backup_20261110_121928/src/company_vision/vision/track_lifecycle_guard.py
reports/step5_fix_20261110_122620/stable_pose_manager.py
reports/step5_retry_20261110_122208/scripts/run_dahua_live_pose.py
reports/step5_retry_20261110_122208/src/company_vision/vision/stable_pose_manager.py
reports/step5_retry_20261110_122208/src/company_vision/vision/track_lifecycle_guard.py
reports/step5_retry_20261110_122455/scripts/run_dahua_live_pose.py
reports/step5_retry_20261110_122455/src/company_vision/vision/stable_pose_manager.py
reports/step5_retry_20261110_122455/src/company_vision/vision/track_lifecycle_guard.py
scripts/analyze_phase4_node42.py
scripts/benchmark_node45_live.before_node45_fix.20261110_173318.py
scripts/benchmark_node45_live.py.before_fixed_duration.py
scripts/benchmark_node45_live.py.before_interrupt_diagnostic.py
scripts/benchmark_node45_live.py.before_interrupt_trace.py
scripts/benchmark_node45_live.py.broken_20261110_190543.py
scripts/benchmark_section1_adaptive.py
scripts/benchmark_section1_diagnostic.py
scripts/benchmark_section1_filters.py
scripts/benchmark_section1_hybrid.py
scripts/benchmark_section1_hybrid_v2.py
scripts/benchmark_section1_metrics.py
scripts/benchmark_section1_stress.py
scripts/diagnose_pose_detections.py
scripts/evaluate_person_validation.py
scripts/inspect_section1_dynamic.py
scripts/profile_phase4_node41.py
scripts/run_dahua_live_compare.py
scripts/run_phase4_node41.ps1
scripts/test_section1_live.py
src/company_vision/vision/async_pose_pipeline.before_node45_fix.20261110_173318.py

- `reports/phase4_node46/cleanup_inventory_20261111_121151.txt` | 7152 bytes | SHA256 `32c2d8303259ae0f25fa947f35dd9014331f3aa2cb267c27810d449f028cf050`

### cleanup_inventory_20261111_121151.txt

p.zip
reports/node45_live_benchmark_20261110_173400.csv
reports/node45_live_benchmark_20261110_173641.csv
reports/node45_live_benchmark_20261110_175147.csv
reports/node45_live_benchmark_20261110_183813.csv
reports/node45_live_benchmark_20261110_184816.csv
reports/node45_live_benchmark_20261110_185229.csv
reports/node45_live_benchmark_20261110_185804.csv
reports/node45_live_benchmark_20261110_190810.csv
reports/node45_live_benchmark_20261110_191104.csv
reports/node45_live_benchmark_20261111_110204.csv
reports/node45_live_benchmark_20261111_110958.csv
reports/node45_live_benchmark_20261111_111248.csv
reports/phase4_final_audit/PHASE4_FINAL_ACCEPTANCE_20261111_120023.md
reports/phase4_final_audit/PHASE4_FINAL_AUDIT_20261111_112211.md
reports/phase4_final_audit/node3_6_final.json
reports/phase4_final_audit/node3_6_smoke.json
reports/phase4_final_audit/preserved_backups/async_pose_pipeline_before_node45_fix_20261111_120023.py
reports/phase4_node41/profile_20261110_125245.json
reports/phase4_node42/backups/run_dahua_live_pose_20261110_130832.py
reports/phase4_node42/backups/run_dahua_live_pose_20261110_130900.py
reports/phase4_node42/capture_analysis_20261110_125639.json
reports/phase4_node42/capture_source_review.txt
reports/phase4_node43/backups/pre_quantize_fix_20261110_134542.py
reports/phase4_node43/backups/run_dahua_live_pose_20261110_132034.py
reports/phase4_node44/backups/run_dahua_live_pose_20261110_135349.py
reports/phase4_node44/copy_20261110_142237.json
reports/phase4_node44/inplace_20261110_142237.json
reports/phase4_node44/node44_final_20261110_144923.json
reports/phase4_node44/reverse_copy_20261110_144923.json
reports/phase4_node44/reverse_inplace_20261110_144020.json
reports/phase4_node44/reverse_inplace_20261110_144310.json
reports/phase4_node44/reverse_inplace_20261110_144720.json
reports/phase4_node44/reverse_inplace_20261110_144923.json
reports/phase4_node44/reverse_inplace_20261110_145550.json
reports/phase4_node45/backups/pre_node45_20261110_160519.py
reports/phase4_node45/backups/run_dahua_live_pose.node44_20261110_183449.py
reports/phase4_node45/backups/run_dahua_live_pose.node44_20261110_183515.py
reports/phase4_node45/backups/run_dahua_live_pose.pre_restore.py
reports/phase4_node45/backups/run_dahua_live_pose_20261110_150936.py
reports/phase4_node45/backups/run_dahua_live_pose_20261110_162706.py
reports/phase4_node45/backups/stable_pose_manager.node44_20261110_183449.py
reports/phase4_node45/backups/stable_pose_manager.node44_20261110_183515.py
reports/phase4_node45/current_context/git_status.txt
reports/phase4_node45/current_context/reports/phase4_node44/node44_final_20261110_144923.json
reports/phase4_node45/current_context/scripts/run_dahua_live_pose.py
reports/phase4_node45/current_context/src/company_vision/vision/one_euro_pose_filter.py
reports/phase4_node45/current_context/src/company_vision/vision/stable_pose_manager.py
reports/phase4_node45/current_context/src/company_vision/vision/track_lifecycle_guard.py
reports/phase4_node45/current_context/working_changes.patch
reports/phase4_node46/NODE46_FINAL_VALIDATION.md
reports/phase4_node46/cleanup_inventory.txt
reports/phase4_node46/node46_final_validation.json
reports/phase4_node46/node46_live_benchmark.csv
reports/pose_diagnostic_20261110_120615/detections.csv
reports/pose_diagnostic_20261110_120615/frames/frame_000001.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000043.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000088.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000131.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000174.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000217.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000261.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000304.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000348.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000392.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000436.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000480.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000524.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000568.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000612.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000656.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000700.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000744.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000786.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000830.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000874.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000917.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_000960.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_001004.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_001048.jpg
reports/pose_diagnostic_20261110_120615/frames/frame_001092.jpg
reports/pose_diagnostic_20261110_120615/r.zip
reports/pose_diagnostic_20261110_120615/validation_evaluation/summary.json
reports/pose_diagnostic_20261110_120615/validation_evaluation/validation_results.csv
reports/step5_backup_20261110_121746/scripts/run_dahua_live_pose.py
reports/step5_backup_20261110_121746/src/company_vision/vision/stable_pose_manager.py
reports/step5_backup_20261110_121746/src/company_vision/vision/track_lifecycle_guard.py
reports/step5_backup_20261110_121928/scripts/run_dahua_live_pose.py
reports/step5_backup_20261110_121928/src/company_vision/vision/stable_pose_manager.py
reports/step5_backup_20261110_121928/src/company_vision/vision/track_lifecycle_guard.py
reports/step5_fix_20261110_122620/stable_pose_manager.py
reports/step5_retry_20261110_122208/scripts/run_dahua_live_pose.py
reports/step5_retry_20261110_122208/src/company_vision/vision/stable_pose_manager.py
reports/step5_retry_20261110_122208/src/company_vision/vision/track_lifecycle_guard.py
reports/step5_retry_20261110_122455/scripts/run_dahua_live_pose.py
reports/step5_retry_20261110_122455/src/company_vision/vision/stable_pose_manager.py
reports/step5_retry_20261110_122455/src/company_vision/vision/track_lifecycle_guard.py
scripts/analyze_phase4_node42.py
scripts/benchmark_node45_live.before_node45_fix.20261110_173318.py
scripts/benchmark_node45_live.py.before_fixed_duration.py
scripts/benchmark_node45_live.py.before_interrupt_diagnostic.py
scripts/benchmark_node45_live.py.before_interrupt_trace.py
scripts/benchmark_node45_live.py.broken_20261110_190543.py
scripts/benchmark_section1_adaptive.py
scripts/benchmark_section1_diagnostic.py
scripts/benchmark_section1_filters.py
scripts/benchmark_section1_hybrid.py
scripts/benchmark_section1_hybrid_v2.py
scripts/benchmark_section1_metrics.py
scripts/benchmark_section1_stress.py
scripts/diagnose_pose_detections.py
scripts/evaluate_person_validation.py
scripts/inspect_section1_dynamic.py
scripts/profile_phase4_node41.py
scripts/run_dahua_live_compare.py
scripts/run_phase4_node41.ps1
scripts/test_section1_live.py

- `reports/phase4_node46/final_live_acceptance_20261111_121151.log` | 3904 bytes | SHA256 `083e9a60ecc79816d4f95498fb6f59e9ead5819af469afe2aec2215cffb6c162`

### final_live_acceptance_20261111_121151.log

��G P U :   N V I D I A   G e F o r c e   R T X   4 0 7 0   S U P E R 
 
 L o a d i n g   Y O L O   P o s e . . . 
 
 P O S E   M O D E L :   y o l o 2 6 x - p o s e . p t 
 
 C a m e r a   U s e r n a m e :   
 
 I n i t i a l i z i n g   D a h u a   S D K . . . 
 
 L O G I N :   P A S S 
 
 C H A N N E L S :   8 
 
 R E A L P L A Y :   P A S S 
 
 Y O L O   P O S E :   R E A D Y 
 
 A C T I V E   K E Y P O I N T S :   H E A D   +   S H O U L D E R S   +   E L B O W S 
 
 A C T I V E   K E Y P O I N T   C O U N T :   9 
 
 W a i t i n g   f o r   c a m e r a   f r a m e . . . 
 
 N O D E   4 . 5 :   O R I G I N A L   S Y N C   P A T H   A C T I V E 
 
 G P U   P R E C I S I O N :   F P 1 6 
 
 A N N O T A T I O N   B U F F E R :   I N _ P L A C E 
 
 
 
 L I V E   Y O L O   P O S E   S T A R T E D 
 
 P r e s s   Q   t o   s t o p . 
 
 F P S = 1 4 . 1 2   D E C O D E D = 1 1 1   P R O C E S S E D = 6 2   P A C K E T S = 1 6 8   P E R S O N S = 2   I N F E R E N C E = 1 8 . 7 m s 
 
 F P S = 1 9 . 8 1   D E C O D E D = 2 3 6   P R O C E S S E D = 1 8 6   P A C K E T S = 3 5 5   P E R S O N S = 1   I N F E R E N C E = 1 8 . 8 m s 
 
 F P S = 2 1 . 4 9   D E C O D E D = 3 6 2   P R O C E S S E D = 3 1 0   P A C K E T S = 5 4 3   P E R S O N S = 2   I N F E R E N C E = 1 8 . 6 m s 
 
 F P S = 2 2 . 4 3   D E C O D E D = 4 8 7   P R O C E S S E D = 4 3 6   P A C K E T S = 7 3 2   P E R S O N S = 2   I N F E R E N C E = 1 9 . 7 m s 
 
 F P S = 2 2 . 9 6   D E C O D E D = 6 1 2   P R O C E S S E D = 5 6 1   P A C K E T S = 9 1 9   P E R S O N S = 2   I N F E R E N C E = 2 0 . 0 m s 
 
 F P S = 2 3 . 3 0   D E C O D E D = 7 3 7   P R O C E S S E D = 6 8 6   P A C K E T S = 1 1 0 7   P E R S O N S = 2   I N F E R E N C E = 1 9 . 0 m s 
 
 F P S = 2 3 . 5 5   D E C O D E D = 8 6 3   P R O C E S S E D = 8 1 2   P A C K E T S = 1 2 9 6   P E R S O N S = 2   I N F E R E N C E = 1 8 . 9 m s 
 
 F P S = 2 3 . 6 6   D E C O D E D = 1 0 4 1   P R O C E S S E D = 8 9 4   P A C K E T S = 1 5 6 9   P E R S O N S = 2   I N F E R E N C E = 2 8 . 8 m s 
 
 F P S = 2 1 . 6 3   D E C O D E D = 1 1 6 6   P R O C E S S E D = 1 0 0 8   P A C K E T S = 1 7 6 4   P E R S O N S = 2   I N F E R E N C E = 3 3 . 5 m s 
 
 F P S = 2 1 . 4 9   D E C O D E D = 1 2 9 1   P R O C E S S E D = 1 1 0 9   P A C K E T S = 1 9 5 6   P E R S O N S = 2   I N F E R E N C E = 2 7 . 8 m s 
 
 F P S = 2 1 . 4 2   D E C O D E D = 1 4 1 6   P R O C E S S E D = 1 2 1 3   P A C K E T S = 2 1 4 8   P E R S O N S = 1   I N F E R E N C E = 2 7 . 5 m s 
 
 F P S = 2 1 . 3 9   D E C O D E D = 1 5 4 1   P R O C E S S E D = 1 3 1 8   P A C K E T S = 2 3 4 2   P E R S O N S = 2   I N F E R E N C E = 2 7 . 0 m s 
 
 F P S = 2 1 . 3 2   D E C O D E D = 1 6 6 6   P R O C E S S E D = 1 4 2 0   P A C K E T S = 2 5 3 4   P E R S O N S = 2   I N F E R E N C E = 2 6 . 1 m s 
 
 F P S = 2 1 . 3 1   D E C O D E D = 1 7 9 2   P R O C E S S E D = 1 5 2 6   P A C K E T S = 2 7 2 7   P E R S O N S = 2   I N F E R E N C E = 2 7 . 3 m s 
 
 F P S = 2 1 . 3 2   D E C O D E D = 1 9 1 7   P R O C E S S E D = 1 6 3 4   P A C K E T S = 2 9 1 7   P E R S O N S = 2   I N F E R E N C E = 1 8 . 5 m s 
 
 
 
 S t o p p i n g   l i v e   p r o c e s s i n g . . . 
 
 
 
 = = =   L I V E   Y O L O   P O S E   R E P O R T   = = = 
 
 P R O C E S S E D   F R A M E S :   1 6 9 4 
 
 E L A P S E D :   7 9 . 0 8 
 
 A V E R A G E   F P S :   2 1 . 4 2 
 
 A V E R A G E   I N F E R E N C E   M S :   2 3 . 6 1 
 
 
 
 = = =   N O D E   4 . 2   C A P T U R E   R E P O R T   = = = 
 
 D E C O D E D   F R A M E S :   1 9 7 7 
 
 S K I P P E D   S E Q U E N C E S :   2 8 3 
 
 C A P T U R E   S A M P L E S :   1 6 9 4 
 
 F R A M E   A G E   M E A N   M S :   9 . 0 2 1 
 
 F R A M E   A G E   M A X   M S :   7 8 . 0 
 
 S D K   I N P U T   E R R O R S :   0 
 
 S D K   C L E A N U P :   C O M P L E T E 
 
 0 1 
 
 0 2 
 
 0 4 
 
 

- `reports/phase4_node46/final_three_actions_20261111_121151.md` | 933 bytes | SHA256 `6fa7433ca102021d4551bdcd986d5a51f962791f293b79b6dc75f754e6339f81`

### final_three_actions_20261111_121151.md

# PHASE 4 - FINAL THREE ACTIONS

Date: 2026-11-11 12:14:51
Branch: feature/phase4-node4-5-async-main
Commit: f053f2a628f0ea35ce89f4861e3051bc5a527a9e

## Pipeline Decision

- Reference pipeline: SYNC
- ASYNC: Not selected as primary pipeline
- Fixed person-count limit: None

## Action 1 - Live Camera

- Duration: 171.6 seconds
- Minimum duration: 60 seconds
- Result: PASS
- Log: reports\phase4_node46\final_live_acceptance_20261111_121151.log

## Action 2 - Detection and Tracking

- Positive person detection: PASS
- Visual tracking: USER CONFIRMED

## Action 3 - Cleanup Audit

- Untracked file count: 117
- Inventory: reports\phase4_node46\cleanup_inventory_20261111_121151.txt
- Automatic deletion: NONE

## Git

- Commit: f053f2a628f0ea35ce89f4861e3051bc5a527a9e
- Automatic commit: NONE
- Automatic push: NONE

## Decision

Three validation actions executed. Awaiting final review.

- `reports/phase4_node46/node46_final_validation.json` | 2508 bytes | SHA256 `09ded0dafe7689f5da7905691a342503dd478e44f25749a5e991e0cf438ff75e`
- `reports/phase4_node46/NODE46_FINAL_VALIDATION.md` | 401 bytes | SHA256 `7f2c43d4ac43277b735cbe04352e28c7fd2a2fa8572b238646ce06a9fa33f5e2`

### NODE46_FINAL_VALIDATION.md

# Node 4.6 - Final Validation

- Camera persons: 2
- Camera detection: PASS
- CUDA / GPU: PASS
- Python syntax: PASS
- Ruff: PASS
- Pytest: 245 passed
- SYNC mean FPS: 24.917
- ASYNC mean FPS: 0
- ASYNC FPS change: -100%
- Benchmark: 4 complete phases
- Protected architecture: unchanged

Phase 5 and Phase 6 were not started.

Acceptance: VALIDATION PASSED.
Final cleanup and Git review pending.

- `reports/phase4_node46/node46_live_benchmark.csv` | 567 bytes | SHA256 `cfa7da0c36c65af7eb7087bb75d20f3e4bf3c01b5d913dd65769f86d5ed945c0`