#!/usr/bin/env python3
"""Test MediaPipe Hands detection independently."""

import cv2
import sys

def test_hand_detection():
    """Test if MediaPipe can detect hands from camera."""
    print("Testing MediaPipe Hands detection...")
    print("-" * 50)
    
    try:
        import mediapipe as mp
        print("✅ MediaPipe imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import MediaPipe: {e}")
        print("   Run: pip install mediapipe")
        return
    
    # Initialize Hands
    print("\nInitializing MediaPipe Hands...")
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("✅ Hands initialized")
    
    # Open camera
    print("\nOpening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return
    
    print("✅ Camera opened")
    
    # Read a few frames
    print("\nCapturing frames (showing your hand to camera)...")
    print("Press 'q' to quit\n")
    
    frame_count = 0
    hand_detected_count = 0
    
    while frame_count < 100:  # Test for ~5 seconds at 20 FPS
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame_count += 1
        
        # Convert and process
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Check results
        if results and results.multi_hand_landmarks:
            hand_detected_count += 1
            num_hands = len(results.multi_hand_landmarks)
            
            # Get hand info
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Check if index finger is extended
                landmarks = hand_landmarks.landmark
                
                # Get index finger tip and PIP
                wrist = landmarks[0]
                index_tip = landmarks[8]
                index_pip = landmarks[6]
                
                # Calculate distances
                def dist_sq(p1, p2):
                    return (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2
                
                tip_to_wrist = dist_sq(index_tip, wrist)
                pip_to_wrist = dist_sq(index_pip, wrist)
                
                ratio = tip_to_wrist / pip_to_wrist if pip_to_wrist > 0 else 0
                index_extended = ratio > 1.1
                
                print(f"Frame {frame_count}: {num_hands} hand(s), Hand {i+1}: Index extended={index_extended:.2f} (ratio={ratio:.2f})")
        else:
            if frame_count % 10 == 0:
                print(f"Frame {frame_count}: No hands detected")
        
        # Show frame
        cv2.imshow("Hand Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Summary
    print("\n" + "-" * 50)
    print(f"Summary: {hand_detected_count}/{frame_count} frames with hand detected")
    
    if hand_detected_count == 0:
        print("\n❌ No hands detected!")
        print("Possible reasons:")
        print("  - Hand not in camera view")
        print("  - Poor lighting conditions")
        print("  - Hand too far/too close to camera")
        print("  - min_detection_confidence too high")
    else:
        print(f"\n✅ Hand detection working ({hand_detected_count/frame_count*100:.1f}% success rate)")
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    test_hand_detection()
