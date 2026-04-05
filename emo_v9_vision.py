#!/usr/bin/env python3
"""emo_v9_vision.py - Reachy Mini Chat v9 with Vision Capabilities

Extends emo_v9.py with computer vision features:
- Face tracking: Robot head follows user's face
- Motion wake-up: Auto-wake when person detected
- Future: Gesture recognition, emotion analysis, visual QA

Usage:
    python emo_v9_vision.py --vision              # Enable vision
    python emo_v9_vision.py --vision --chat       # Interactive chat with vision
    python emo_v9_vision.py --no-vision           # Disable vision (pure v9 mode)

Architecture:
    This file extends ChatAppWithPiper from emo_v9.py, adding VisionController
    as an optional plugin. The original v9 code remains untouched.
"""

import sys
import time
import argparse
import threading
from typing import Optional

# Import base v9 functionality
from emo_v9 import (
    ChatAppWithPiper,
    EmotionControllerV71,
    ConversationHistory,
    PiperTTSEngine,
    LipSyncControllerV5,
)

# Import vision module
try:
    from vision import VisionController, VisionConfig
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
        vision_fps: float = 15.0,
        vision_auto_wake: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.vision_enabled = vision_enabled and VISION_AVAILABLE
        self.vision: Optional[VisionController] = None
        self._vision_config = VisionConfig(
            enabled=self.vision_enabled,
            face_tracking=True,
            target_fps=vision_fps,
            auto_wake=vision_auto_wake,
            track_while_speaking=True
        )
        
        # State for vision integration
        self._person_present = False
        self._face_tracking_active = False
    
    async def start_chat_async(self):
        """Start chat with vision capabilities."""
        print("=" * 60)
        print("🤖 Reachy Mini Chat v9 with Vision")
        print("=" * 60)
        
        if self.vision_enabled:
            print("👁️  Vision features: ENABLED")
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
            # Options: "default", "opencv", "gstreamer", "no_media"
            media_backend = "default" if self.vision_enabled else "no_media"
            with ReachyMini(media_backend=media_backend) as reachy:
                print("✅ Connected to Reachy Mini")
                
                # Disable automatic body yaw for recorded moves
                reachy.set_automatic_body_yaw(False)
                
                # Initialize vision controller
                if self.vision_enabled:
                    self.vision = VisionController(reachy, self._vision_config)
                    self._setup_vision_callbacks()
                    self.vision.start()
                
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
                
                # Enter main chat loop
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
    
    def _setup_vision_callbacks(self):
        """Setup vision event callbacks."""
        if not self.vision:
            return
        
        # When face detected: can be used for immediate reactions
        def on_face(pos):
            pass  # Face position used in animation loop
        
        # When person enters frame: wake up if sleeping
        def on_person_enter():
            if self._vision_config.auto_wake and not self._person_present:
                print("   👋 Person detected!")
                self._person_present = True
                # Could trigger wake animation here
        
        # When person leaves: could enter sleep mode
        def on_person_leave():
            self._person_present = False
            print("   😴 No person detected")
        
        self.vision.on_face_detected = on_face
        self.vision.on_person_enter = on_person_enter
        self.vision.on_person_leave = on_person_leave
    
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
                    
                    # Periodically look at face (every 1.5 seconds)
                    current_time = time.time()
                    if current_time - last_face_look > 1.5:
                        if self.vision and self.vision.is_person_present():
                            if pos := self.vision.get_face_position():
                                try:
                                    self.controller.reachy.look_at_image(
                                        pos[0], pos[1], duration=0.3
                                    )
                                    print(f"   👁️  Looking at face ({pos[0]}, {pos[1]})")
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
        
        # Start animation thread
        anim_thread = threading.Thread(target=animation_thread, daemon=True)
        anim_thread.start()
        
        # Run TTS
        speak_result = self.controller.tts_engine.speak_with_interrupt(
            text, emotion=emotion, stop_event=stop_event
        )
        
        tts_done.set()
        anim_thread.join(timeout=20.0)
        
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
        """ASR chat mode with vision."""
        # Use parent implementation but with vision-aware speak method
        # For now, delegate to parent and override speak method
        self._override_speak_method()
        await super().start_chat_async()
    
    async def _chat_text(self, reachy):
        """Text chat mode with vision."""
        self._override_speak_method()
        await super().start_chat_async()
    
    def _override_speak_method(self):
        """Temporarily override speak method to include vision."""
        # Store original
        self._original_speak = self._speak_and_animate
        # Replace with vision version
        self._speak_and_animate = self._speak_and_animate_with_vision


def main():
    parser = argparse.ArgumentParser(
        description="Reachy Mini Chat v9 with Vision"
    )
    
    # Vision arguments
    parser.add_argument(
        '--vision', 
        action='store_true',
        default=True,
        help='Enable vision features (default: True)'
    )
    parser.add_argument(
        '--no-vision',
        action='store_true',
        help='Disable vision features'
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
    
    # Determine vision enabled state
    vision_enabled = args.vision and not args.no_vision
    
    if args.no_vision:
        vision_enabled = False
    
    # Create app
    app = ChatAppWithVision(
        vision_enabled=vision_enabled,
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
