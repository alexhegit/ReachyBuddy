#!/usr/bin/env python3
"""emo_v9_vision.py - Reachy Mini Chat v9 with Vision Capabilities

Extends emo_v9.py with computer vision features:
- Face tracking: Robot head follows user's face
- Security monitor: Motion detection and person presence logging
- Motion wake-up: Auto-wake when person detected

Usage:
    python emo_v9_vision.py                           # Pure v9 mode (no vision)
    python emo_v9_vision.py --vision face --chat      # Enable face tracking
    python emo_v9_vision.py --vision monitor          # Security monitoring mode

Architecture:
    This file extends ChatAppWithPiper from emo_v9.py, adding VisionController
    as an optional plugin. The original v9 code remains untouched.
"""

import sys
import time
import argparse
import threading
import asyncio
from typing import Optional

# Import base v9 functionality
from emo_v9 import (
    ChatAppWithPiper,
    EmotionControllerV71,
    ConversationHistory,
    PiperTTSEngine,
    LipSyncControllerV5,
    FasterWhisperASREngine,
)

# Import vision module
try:
    from vision import VisionController, VisionConfig, MonitorTracker, FaceTracker
    VISION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Vision module not available: {e}")
    print("   Run: pip install mediapipe opencv-python")
    VISION_AVAILABLE = False

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


class ChatAppWithVision(ChatAppWithPiper):
    """Chat application with vision capabilities.
    
    Extends ChatAppWithPiper to add face tracking and visual interaction.
    Vision features are optional and controlled via --vision flag.
    
    Args:
        vision_enabled: Master switch for vision features
        vision_fps: Target camera processing FPS
        vision_auto_wake: Wake robot when person detected
        *args, **kwargs: Passed to ChatAppWithPiper
    """
    
    def __init__(
        self,
        vision_enabled: bool = True,
        vision_mode: Optional[str] = 'face',
        vision_fps: float = 15.0,
        vision_auto_wake: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.vision_enabled = vision_enabled and VISION_AVAILABLE
        self.vision_mode = vision_mode
        self.enable_face = vision_mode == 'face'  # Face tracking enabled in 'face' mode
        
        self.vision: Optional[VisionController] = None
        self.monitor_tracker: Optional[MonitorTracker] = None
        
        self._vision_config = VisionConfig(
            enabled=self.vision_enabled and vision_mode == 'face',
            face_tracking=True,
            target_fps=vision_fps,
            auto_wake=vision_auto_wake,
            track_while_speaking=True
        )
        
        # State for vision integration
        self._person_present = False
        self._face_tracking_active = False
        self._is_speaking = False  # Track speaking state for idle face tracking
        self._monitor_active = False  # Track if monitor mode is active
        self._patrol_active = False  # Track if patrol is active
        self._patrol_paused = False  # Pause patrol on events
    
    async def start_chat_async(self):
        """Start chat with vision capabilities."""
        print("=" * 60)
        print("🤖 Reachy Mini Chat v9 with Vision")
        print("=" * 60)
        
        if self.vision_enabled:
            if self.vision_mode == 'monitor':
                print(f"👁️  Vision mode: SECURITY MONITOR 🔒")
                print(f"   - Motion detection: Yes")
                print(f"   - Person presence: Yes")
                print(f"   - Event logging: Yes")
            else:
                print(f"👁️  Vision features: ENABLED")
                print(f"   - Face tracking: Yes")
                print(f"   - Target FPS: {self._vision_config.target_fps}")
                print(f"   - Auto wake: {self._vision_config.auto_wake}")
        else:
            print("👁️  Vision features: DISABLED")
            if not VISION_AVAILABLE:
                print("   (MediaPipe not installed)")
        
        print(f"🎙️  Piper Model: {self.piper_model}")
        print(f"💬 History: {self.history.max_rounds} rounds")
        print("-" * 60)
        
        try:
            # Use "default" media backend to enable camera for vision
            media_backend = "default" if self.vision_enabled else "no_media"
            
            # For monitor mode, we need camera but not face tracking config
            if self.vision_mode == 'monitor':
                # Create minimal vision config for camera access
                self._vision_config = VisionConfig(enabled=True, face_tracking=True)
            
            with ReachyMini(media_backend=media_backend) as reachy:
                print("✅ Connected to Reachy Mini")
                
                # Disable automatic body yaw for recorded moves
                reachy.set_automatic_body_yaw(False)
                
                # Initialize vision controller based on mode
                if self.vision_enabled:
                    if self.vision_mode == 'monitor':
                        # Security monitor mode - no chat, just monitoring
                        self._start_security_monitoring(reachy)
                        
                        # Keep running until interrupted
                        print("\n   🔒 Monitor mode active. Press Ctrl+C to stop.\n")
                        try:
                            while self._monitor_active:
                                await self._sleep(1.0)
                        except KeyboardInterrupt:
                            print("\n   🛑 Monitor stopped by user")
                        return
                    else:
                        # Face tracking mode (default)
                        self.vision = VisionController(reachy, self._vision_config)
                        self._setup_vision_callbacks()
                        self.vision.start()
                        self._start_idle_face_tracking(reachy)
                
                # Initialize emotion controller (from v9)
                self.controller = EmotionControllerV71(
                    reachy,
                    self.piper_model,
                    self.piper_config,
                    self.speaker_id,
                    self.debug,
                    gentle_mode=self.gentle
                )
                
                # Initial pose
                reachy.goto_target(head=create_head_pose(), duration=1.0)
                await self._sleep(1.0)
                
                # Enter main chat loop (only for non-monitor modes)
                if self.use_asr:
                    await self._chat_with_asr(reachy)
                else:
                    await self._chat_text(reachy)
                    
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
        finally:
            if self.vision:
                self.vision.stop()
            if self.monitor_tracker:
                self._monitor_active = False
                self._patrol_active = False
                self.monitor_tracker.stop()
    
    def _setup_vision_callbacks(self):
        """Setup vision event callbacks."""
        if not self.vision:
            return
        
        # When person enters frame: wake up if sleeping
        def on_person_enter():
            if self._vision_config.auto_wake and not self._person_present:
                print("   👋 Person detected!")
                self._person_present = True
        
        # When person leaves: could enter sleep mode
        def on_person_leave():
            self._person_present = False
            print("   😴 No person detected")
        
        self.vision.on_person_enter = on_person_enter
        self.vision.on_person_leave = on_person_leave
    
    def _start_idle_face_tracking(self, reachy):
        """Start background thread for ultra-smooth face tracking.
        
        Prioritizes smoothness over low latency for natural movement.
        """
        import threading
        
        def idle_tracker():
            """Ultra-smooth face tracking when idle."""
            print("   👁️  Idle face tracking started (ultra-smooth)")
            
            # Triple EMA for maximum smoothness
            # Layer 1: Input smoothing
            ema1_x: Optional[float] = None
            ema1_y: Optional[float] = None
            alpha1 = 0.3
            
            # Layer 2: Secondary smoothing  
            ema2_x: Optional[float] = None
            ema2_y: Optional[float] = None
            alpha2 = 0.2
            
            # Layer 3: Final output smoothing
            ema3_x: Optional[float] = None
            ema3_y: Optional[float] = None
            alpha3 = 0.15
            
            last_sent_pos: Optional[Tuple[int, int]] = None
            min_update_interval = 0.08  # 12.5 FPS - lower frequency for smoother motion
            position_threshold = 40  # Higher threshold = fewer updates = smoother
            
            last_update_time = 0.0
            
            while self.vision and self.vision._running:
                current_time = time.time()
                
                if current_time - last_update_time < min_update_interval:
                    time.sleep(0.01)
                    continue
                
                if not self._is_speaking:
                    if self.vision.is_person_present():
                        if pos := self.vision.get_face_position():
                            raw_x, raw_y = pos
                            
                            # Triple EMA cascade
                            if ema1_x is None:
                                ema1_x, ema1_y = float(raw_x), float(raw_y)
                                ema2_x, ema2_y = ema1_x, ema1_y
                                ema3_x, ema3_y = ema2_x, ema2_y
                            else:
                                # Layer 1
                                ema1_x = alpha1 * raw_x + (1 - alpha1) * ema1_x
                                ema1_y = alpha1 * raw_y + (1 - alpha1) * ema1_y
                                # Layer 2
                                ema2_x = alpha2 * ema1_x + (1 - alpha2) * ema2_x
                                ema2_y = alpha2 * ema1_y + (1 - alpha2) * ema2_y
                                # Layer 3 (output)
                                ema3_x = alpha3 * ema2_x + (1 - alpha3) * ema3_x
                                ema3_y = alpha3 * ema2_y + (1 - alpha3) * ema3_y
                            
                            final_pos = (int(ema3_x), int(ema3_y))
                            
                            # Higher threshold = fewer movements = smoother
                            should_update = True
                            if last_sent_pos:
                                dx = abs(final_pos[0] - last_sent_pos[0])
                                dy = abs(final_pos[1] - last_sent_pos[1])
                                if dx < position_threshold and dy < position_threshold:
                                    should_update = False
                            
                            if should_update:
                                try:
                                    # Longer duration for very smooth movement
                                    reachy.look_at_image(
                                        final_pos[0], final_pos[1], 
                                        duration=0.6  # Longer = smoother
                                    )
                                    last_sent_pos = final_pos
                                    last_update_time = current_time
                                    if self.debug:
                                        print(f"   👁️  Track ({raw_x},{raw_y})→({final_pos[0]},{final_pos[1]})")
                                except Exception:
                                    pass
                    else:
                        if last_sent_pos is not None:
                            try:
                                reachy.goto_target(head=create_head_pose(), duration=0.8)
                                last_sent_pos = None
                                ema1_x = ema1_y = None
                                ema2_x = ema2_y = None
                                ema3_x = ema3_y = None
                                last_update_time = current_time
                                if self.debug:
                                    print("   👁️  No face - center")
                            except Exception:
                                pass
                
                time.sleep(0.01)
        
        tracker_thread = threading.Thread(target=idle_tracker, daemon=True)
        tracker_thread.start()
        print("   ✅ Ultra-smooth face tracking started")
    
    def _start_security_monitoring(self, reachy):
        """Start security monitoring mode.
        
        Monitors for motion, person presence, and anomalies.
        Logs events with timestamps.
        """
        import threading
        
        self.monitor_tracker = MonitorTracker(
            motion_threshold=25,
            min_motion_area=500,
            cooldown_seconds=5.0,
            buffer_seconds=3.0
        )
        self.monitor_tracker.start()
        self._monitor_active = True
        
        # Setup event callbacks
        def on_motion_event(event):
            print(f"   🚨 MOTION DETECTED at {event.timestamp.strftime('%H:%M:%S')}")
            print(f"      {event.description}")
            # Optional: trigger robot to look at motion direction
            try:
                reachy.goto_target(head=create_head_pose(), duration=0.5)
            except Exception:
                pass
        
        def on_person_enter_event(event):
            print(f"   👤 PERSON ENTERED at {event.timestamp.strftime('%H:%M:%S')}")
            print(f"      {event.description}")
            # Person detected - robot looks around alertly
            try:
                reachy.goto_target(head=create_head_pose(yaw=15, degrees=True), duration=0.3)
                time.sleep(0.3)
                reachy.goto_target(head=create_head_pose(yaw=-15, degrees=True), duration=0.3)
                time.sleep(0.3)
                reachy.goto_target(head=create_head_pose(), duration=0.3)
            except Exception:
                pass
        
        def on_person_leave_event(event):
            print(f"   🚪 PERSON LEFT at {event.timestamp.strftime('%H:%M:%S')}")
            print(f"      {event.description}")
        
        def on_anomaly_event(event):
            print(f"   ⚠️  ANOMALY at {event.timestamp.strftime('%H:%M:%S')}")
            print(f"      {event.description}")
        
        self.monitor_tracker.on_motion = on_motion_event
        self.monitor_tracker.on_person_enter = on_person_enter_event
        self.monitor_tracker.on_person_leave = on_person_leave_event
        self.monitor_tracker.on_anomaly = on_anomaly_event
        
        # Create face tracker for person detection in monitor mode
        face_tracker = FaceTracker(smooth_factor=0.3)
        
        # Patrol state
        self._patrol_active = True
        self._patrol_paused = False  # Pause patrol when event detected
        
        def patrol_loop():
            """Head patrol loop: specific waypoints with 1s hold at each position.
            
            Trajectory: Center → Left25° (1s) → Left50° (1s) → Left25° (1s) → Center (1s)
                      → Right25° (1s) → Right50° (1s) → Right25° (1s) → Center (1s) → repeat
            """
            print("   🔄 Head patrol started: Center ↔ Left50° ↔ Right50°")
            
            # Define patrol waypoints: (yaw_angle, hold_time_seconds)
            waypoints = [
                (25, 1.0),    # Left 25°, hold 1s
                (50, 1.0),    # Left 50°, hold 1s
                (25, 1.0),    # Left 25°, hold 1s
                (0, 1.0),     # Center, hold 1s
                (-25, 1.0),   # Right 25°, hold 1s
                (-50, 1.0),   # Right 50°, hold 1s
                (-25, 1.0),   # Right 25°, hold 1s
                (0, 1.0),     # Center, hold 1s
            ]
            
            # Start from center
            try:
                reachy.goto_target(
                    head=create_head_pose(yaw=0, degrees=True),
                    duration=0.5
                )
                time.sleep(0.5)
            except Exception:
                pass
            
            waypoint_index = 0
            
            while self._patrol_active:
                if self._patrol_paused:
                    time.sleep(0.1)
                    continue
                
                try:
                    # Get current waypoint
                    yaw_angle, hold_time = waypoints[waypoint_index]
                    
                    # Move to position (0.4s transition)
                    reachy.goto_target(
                        head=create_head_pose(yaw=yaw_angle, degrees=True),
                        duration=0.4
                    )
                    
                    # Wait for movement + hold time
                    time.sleep(0.4 + hold_time)
                    
                    # Next waypoint
                    waypoint_index = (waypoint_index + 1) % len(waypoints)
                    
                except Exception as e:
                    if self.debug:
                        print(f"      ⚠️ Patrol error: {e}")
                    time.sleep(0.5)
            
            print("   🛑 Head patrol stopped")
        
        def monitor_loop():
            """Continuous monitoring loop."""
            print("   🔒 Monitor loop started")
            frame_count = 0
            
            while self._monitor_active and self.monitor_tracker:
                # Get frame from camera via reachy media
                try:
                    frame = None
                    if hasattr(reachy, 'media') and reachy.media:
                        frame = reachy.media.get_frame()
                    
                    if frame is not None:
                        frame_count += 1
                        
                        # Check for person using face tracker
                        person_detected = False
                        face_pos = face_tracker.get_face_center(frame)
                        person_detected = face_pos is not None
                        
                        # Process frame for monitoring
                        event = self.monitor_tracker.process_frame(frame, person_detected)
                        
                        # Pause patrol on significant event, resume after cooldown
                        if event and event.event_type in ('motion', 'person_enter'):
                            self._patrol_paused = True
                            # Resume patrol after 5 seconds
                            threading.Timer(5.0, lambda: setattr(self, '_patrol_paused', False)).start()
                        
                        # Print stats every 300 frames (~30 seconds at 10 FPS)
                        if frame_count % 300 == 0:
                            stats = self.monitor_tracker.get_event_stats()
                            print(f"\n   📊 Monitor Stats (last 30s):")
                            print(f"      Total events: {stats['total_events']}")
                            print(f"      Motion: {stats['motion_count']}")
                            print(f"      Person enter: {stats['person_enter_count']}")
                            print(f"      Person leave: {stats['person_leave_count']}")
                            print(f"      Anomalies: {stats['anomaly_count']}")
                
                except Exception as e:
                    if self.debug:
                        print(f"      ⚠️ Monitor error: {e}")
                
                time.sleep(0.1)  # 10 FPS monitoring
        
        # Start both threads
        patrol_thread = threading.Thread(target=patrol_loop, daemon=True)
        patrol_thread.start()
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
        print("   ✅ Security monitoring + patrol active")
    
    def _speak_and_animate_with_vision(
        self,
        response: str,
        emotion: str,
        intensity: str,
        emotion_level: float,
        stop_event: threading.Event = None
    ) -> bool:
        """Override speak_and_animate to include face tracking.
        
        This extends the base v9 animation with face tracking:
        - While speaking, robot periodically looks at user's face
        - Maintains eye contact for more natural interaction
        """
        if not self.controller:
            return True
        
        # Use base v9 speak with interrupt
        # But inject face tracking into animation thread
        return self._speak_with_face_tracking(
            response, emotion, intensity, emotion_level, stop_event
        )
    
    def _speak_with_face_tracking(
        self,
        text: str,
        emotion: str,
        intensity: str,
        emotion_level: float,
        stop_event: threading.Event
    ) -> bool:
        """Speak with face tracking enabled."""
        import threading
        
        # Mark as speaking - this pauses idle face tracking
        self._is_speaking = True
        
        print(f"🎙️ Speaking: '{text[:50]}...'")
        
        # Duration setup (same as v9)
        duration_map = {'high': 0.8, 'medium': 1.0, 'low': 1.2}
        if self.controller.gentle_mode:
            duration_map = {'high': 1.0, 'medium': 1.3, 'low': 1.5}
        base_move_duration = duration_map.get(intensity, 1.0)
        
        tts_done = threading.Event()
        
        def animation_thread():
            """Animation loop with face tracking."""
            try:
                emotion_level = 0.5 if emotion == 'neutral' else 0.8
                
                if self.controller.gentle_mode:
                    print("   😌 Gentle mode with face tracking")
                else:
                    print("   🎵 Animation + face tracking")
                
                # Start lip sync
                self.controller.lip_sync.start_lip_sync(text, emotion_level)
                
                last_move = None
                used_moves = set()
                move_counter = 0
                last_face_look = 0.0
                
                while not tts_done.is_set():
                    import random
                    
                    # During speaking: occasional glance at face (if face tracking enabled)
                    if self.enable_face:
                        current_time = time.time()
                        if current_time - last_face_look > 3.0:  # Every 3 seconds while speaking
                            if self.vision and self.vision.is_person_present():
                                if pos := self.vision.get_face_position():
                                    try:
                                        # Quick glance (0.2s) - doesn't interrupt actions much
                                        self.controller.reachy.look_at_image(
                                            pos[0], pos[1], duration=0.2
                                        )
                                        print(f"   👁️  Glance at face ({pos[0]}, {pos[1]})", flush=True)
                                    except Exception:
                                        pass
                            last_face_look = current_time
                    
                    # Continue with normal animation (from v9)
                    roll = random.randint(0, 99)
                    
                    if roll < 50:
                        move, _, speed = self.controller._choose_animation_for_emotion(
                            emotion, intensity, avoid_move=last_move, used_moves=used_moves
                        )
                        if move:
                            last_move = move
                            used_moves.add(move)
                            move_duration = base_move_duration / speed
                            if not self.controller.gentle_mode:
                                print(f"   🎬 {move} ({move_duration:.1f}s)")
                            else:
                                print(f"   🎬 Gentle: {move}")
                            self.controller._play_recorded_move(move, move_duration)
                        else:
                            self.controller._simple_nod_once()
                            time.sleep(0.8)
                    elif roll < 75:
                        if not self.controller.gentle_mode:
                            print("   🎭 Combined action")
                            self.controller._execute_random_combined_action(emotion)
                        else:
                            self.controller._simple_thoughtful_tilt_once()
                            time.sleep(0.8)
                    else:
                        print("   🔄 Body turn")
                        try:
                            angle = random.choice([-0.5, -0.25, 0.25, 0.5])
                            head_tilt = random.choice([
                                create_head_pose(),
                                create_head_pose(roll=10, degrees=True),
                                create_head_pose(roll=-10, degrees=True),
                            ])
                            self.controller.reachy.goto_target(
                                head=head_tilt, body_yaw=angle, duration=0.4
                            )
                            time.sleep(0.45)
                            self.controller.reachy.goto_target(
                                head=create_head_pose(), body_yaw=0.0, duration=0.4
                            )
                            time.sleep(0.45)
                        except Exception:
                            pass
                    
                    move_counter += 1
                    if not tts_done.is_set():
                        time.sleep(0.3)
                
                self.controller.lip_sync.stop_lip_sync()
                print("   ✅ Animation completed")
                
            except Exception as e:
                print(f"⚠️ Animation error: {e}")
                import traceback
                traceback.print_exc()
                self.controller.lip_sync.stop_lip_sync()
                # Ensure speaking flag is cleared on error
                self._is_speaking = False
        
        # Start animation thread
        anim_thread = threading.Thread(target=animation_thread, daemon=True)
        anim_thread.start()
        
        # Run TTS
        speak_result = self.controller.tts_engine.speak_with_interrupt(
            text, emotion=emotion, stop_event=stop_event
        )
        
        tts_done.set()
        anim_thread.join(timeout=20.0)
        
        # Mark as not speaking - resume idle face tracking
        self._is_speaking = False
        
        # Reset body
        try:
            self.controller.reachy.goto_target(body_yaw=0.0, duration=0.5)
        except Exception:
            pass
        
        return speak_result
    
    async def _sleep(self, duration: float):
        """Async sleep helper."""
        import asyncio
        await asyncio.sleep(duration)
    
    async def _chat_with_asr(self, reachy):
        """ASR chat mode with vision - directly implement to avoid double init."""
        # Import here to avoid circular dependency
        import aiohttp
        import select
        import sys
        
        print("\n🎤 VAD ASR + Vision mode: press Ctrl-C to stop")
        
        if FasterWhisperASREngine is None:
            print("❌ ASR not available")
            return
            
        # Initialize ASR
        print(f"Initializing ASR ({self.asr_model}, VAD: {self.vad_silence}s silence)...")
        try:
            self.asr_engine = await asyncio.to_thread(
                FasterWhisperASREngine,
                model_name=self.asr_model,
                device='cpu'
            )
        except Exception as e:
            print(f"❌ Failed to initialize ASR: {e}")
            return
        
        async with aiohttp.ClientSession() as session:
            await self.check_ollama_model(session)
            
            while True:
                try:
                    print("\n🎙️ Speak now... (Ctrl+C to exit)")
                    
                    # Record with VAD
                    asr_start = time.time()
                    
                    if self.use_vad:
                        transcription = await asyncio.to_thread(
                            self.asr_engine.transcribe_from_mic_vad,
                            max_duration=4.0,
                            silence_threshold=self.vad_silence,
                            aggressiveness=self.vad_aggressive,
                            trailing_buffer_ms=300,
                            show_volume=True
                        )
                    else:
                        transcription = await asyncio.to_thread(
                            self.asr_engine.transcribe_from_mic,
                            duration=4.0,
                            show_volume=True
                        )
                    
                    asr_time = time.time() - asr_start
                    
                    if not transcription:
                        print("⚠️ No speech detected, try again")
                        continue
                    
                    # Process user input
                    self.history.add_user_message(transcription)
                    print(f"📝 You: {transcription}")
                    
                    # Get LLM response
                    print("\n🤖 Reachy Mini: ", end="", flush=True)
                    llm_start = time.time()
                    
                    response = await self._get_ollama_response_async(transcription, session)
                    llm_time = time.time() - llm_start
                    
                    if response and self.controller:
                        # Analyze and speak with vision
                        emotion, intensity, emotion_level = self.controller.analyze_emotion(response)
                        self._stop_speaking_event.clear()
                        
                        tts_start = time.time()
                        speech_task = asyncio.create_task(asyncio.to_thread(
                            self._speak_with_face_tracking,
                            response, emotion, intensity, emotion_level,
                            self._stop_speaking_event
                        ))
                        
                        # Wait for interrupt
                        try:
                            while not speech_task.done():
                                if select.select([sys.stdin], [], [], 0)[0]:
                                    try:
                                        char = sys.stdin.read(1)
                                        if char == '':
                                            print("\n⏹️ Interrupting...")
                                            self._stop_speaking_event.set()
                                            break
                                    except:
                                        pass
                                await asyncio.sleep(0.05)
                            
                            await speech_task
                        except asyncio.CancelledError:
                            pass
                        
                        tts_time = time.time() - tts_start
                        self.history.add_assistant_message(response)
                        
                        if self.debug:
                            print(f"\n  ⏱️ ASR: {asr_time:.2f}s, LLM: {llm_time:.2f}s, TTS: {tts_time:.2f}s")
                        
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    return
                except Exception as e:
                    print(f"\n⚠️ Error: {e}")
                    await asyncio.sleep(1.0)
    
    async def _chat_text(self, reachy):
        """Text chat mode with vision - directly implement to avoid double init."""
        import aiohttp
        import select
        import sys
        
        print("\n💬 Start chatting (type 'quit' or Ctrl+C to exit)")
        
        async with aiohttp.ClientSession() as session:
            await self.check_ollama_model(session)
            
            while True:
                try:
                    user_input = input("\n🧑 You: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        break
                    if user_input.lower() == 'clear':
                        self.history.clear()
                        continue
                    if not user_input:
                        continue
                    
                    self.history.add_user_message(user_input)
                    print("\n🤖 Reachy Mini: ", end="", flush=True)
                    
                    # Get LLM response
                    llm_start = time.time()
                    response = await self._get_ollama_response_async(user_input, session)
                    llm_time = time.time() - llm_start
                    
                    if response and self.controller:
                        emotion, intensity, emotion_level = self.controller.analyze_emotion(response)
                        self._stop_speaking_event.clear()
                        
                        tts_start = time.time()
                        speech_task = asyncio.create_task(asyncio.to_thread(
                            self._speak_with_face_tracking,
                            response, emotion, intensity, emotion_level,
                            self._stop_speaking_event
                        ))
                        
                        # Wait for interrupt
                        try:
                            while not speech_task.done():
                                if select.select([sys.stdin], [], [], 0)[0]:
                                    try:
                                        char = sys.stdin.read(1)
                                        if char == '':
                                            print("\n⏹️ Interrupting...")
                                            self._stop_speaking_event.set()
                                            break
                                    except:
                                        pass
                                await asyncio.sleep(0.05)
                            
                            await speech_task
                        except asyncio.CancelledError:
                            pass
                        
                        tts_time = time.time() - tts_start
                        self.history.add_assistant_message(response)
                        
                        if self.debug:
                            print(f"\n  ⏱️ LLM: {llm_time:.2f}s, TTS: {tts_time:.2f}s")
                            
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    return
                except Exception as e:
                    print(f"\n⚠️ Error: {e}")
                    await asyncio.sleep(1.0)


def main():
    parser = argparse.ArgumentParser(
        description="Reachy Mini Chat v9 with Vision"
    )
    
    # Vision arguments
    parser.add_argument(
        '--vision', 
        nargs='?',
        const='face',
        default=None,
        choices=['face', 'monitor'],
        help='Enable vision features: face (face tracking), monitor (security monitoring). Default: disabled'
    )
    parser.add_argument(
        '--vision-fps',
        type=float,
        default=15.0,
        help='Vision processing FPS (default: 15)'
    )
    parser.add_argument(
        '--no-auto-wake',
        action='store_true',
        help='Disable auto-wake on person detection'
    )
    
    # Include base v9 arguments
    parser.add_argument('--model', default="qwen3:0.6b", help='Ollama model')
    parser.add_argument('--ollama-url', default="http://localhost:11434")
    parser.add_argument('--piper-model', default="models/en-us-ryan-medium.onnx")
    parser.add_argument('--piper-config', default=None)
    parser.add_argument('--speaker-id', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--asr', action='store_true', help='Enable ASR')
    parser.add_argument('--gentle', action='store_true')
    parser.add_argument('--history-size', type=int, default=5)
    parser.add_argument('--no-history', action='store_true')
    parser.add_argument('--asr-model', default="small")
    parser.add_argument('--vad-silence', type=float, default=0.8)
    parser.add_argument('--vad-aggressive', type=int, default=1)
    parser.add_argument('--no-vad', action='store_true')
    parser.add_argument('--chat', action='store_true', help='Start interactive chat')
    
    args = parser.parse_args()
    
    # Determine vision mode
    # Default is None (disabled), --vision enables it
    vision_enabled = args.vision is not None
    vision_mode = args.vision  # 'face' or 'monitor'
    enable_face = vision_mode == "face"
    enable_monitor = vision_mode == "monitor"
    
    # Print vision mode info
    if vision_enabled:
        if enable_monitor:
            print(f"🔒 Security monitor mode: ENABLED")
            print(f"   Logs will be saved to: security_logs/")
        else:
            print(f"👁️  Vision features: ENABLED (face tracking)")
    
    # Create app
    app = ChatAppWithVision(
        vision_enabled=vision_enabled,
        vision_mode=vision_mode,
        vision_fps=args.vision_fps,
        vision_auto_wake=not args.no_auto_wake,
        model=args.model,
        ollama_url=args.ollama_url,
        piper_model=args.piper_model,
        piper_config=args.piper_config,
        speaker_id=args.speaker_id,
        debug=args.debug,
        use_asr=args.asr,
        gentle=args.gentle,
        history_size=args.history_size,
        enable_history=not args.no_history,
        asr_model=args.asr_model,
        vad_silence=args.vad_silence,
        vad_aggressive=args.vad_aggressive,
        use_vad=not args.no_vad
    )
    
    import asyncio
    asyncio.run(app.start_chat_async())


if __name__ == '__main__':
    main()
