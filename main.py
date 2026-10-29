import os
import time
import cv2
import pygame
import mediapipe as mp

# --- Configuration ---
MODEL_PATH = "hand_landmarker.task"
SOUNDS_DIR = "sound_swaras"

# --- Initialize Pygame Mixer for Audio (Milestone 7) ---
pygame.mixer.init()

# --- Map Counts to Sound Files (Milestone 6) ---
sounds_mapping = {
    1: os.path.join(SOUNDS_DIR, "1_Sa.mp3"),
    2: os.path.join(SOUNDS_DIR, "2_Re.mp3"),
    3: os.path.join(SOUNDS_DIR, "3_Ga.mp3"),
    4: os.path.join(SOUNDS_DIR, "4_Ma.mp3"),
    5: os.path.join(SOUNDS_DIR, "5_Pa.mp3"),
    6: os.path.join(SOUNDS_DIR, "6_Dha.mp3"),
    7: os.path.join(SOUNDS_DIR, "7_Ni.mp3"),
    8: os.path.join(SOUNDS_DIR, "8_Sa_upper.mp3"),
}

# --- Initialize MediaPipe Hand Landmarker ---
base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,
)
landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

# --- Initialize Webcam ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
  print("Error: Could not open webcam.")
  exit()

start_time = time.time()
previous_count = -1  # Tracks the last played count for trigger-on-change logic

print("Gesture Swara Player running. Press 'q' to quit.")

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    break

  # 1. Mirror the frame horizontally
  frame = cv2.flip(frame, 1)
  h, w, _ = frame.shape

  # 2. Prepare frame for MediaPipe
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

  # 3. Compute timestamp
  timestamp_ms = int((time.time() - start_time) * 1000)

  # 4. Detect hands
  result = landmarker.detect_for_video(mp_image, timestamp_ms)

  total_finger_count = 0

  if result.hand_landmarks:
    for i, hand_landmarks in enumerate(result.hand_landmarks):
      label = result.handedness[i][0].category_name
      hand_count = 0

      # Thumb Logic
      thumb_tip_x = hand_landmarks[4].x
      thumb_joint_x = hand_landmarks[3].x
      if label == "Right":
        if thumb_tip_x < thumb_joint_x:
          hand_count += 1
      else:
        if thumb_tip_x > thumb_joint_x:
          hand_count += 1

      # Four Fingers Logic
      finger_tips = [8, 12, 16, 20]
      finger_pips = [6, 10, 14, 18]
      for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks[tip].y < hand_landmarks[pip].y:
          hand_count += 1

      total_finger_count += hand_count

      # Draw hand landmarks
      for lm in hand_landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

  # 5. Audio Trigger Logic (Trigger on Change)
  # Only play a sound if the count changes and exists in our mapping dictionary (1 to 8)
  if total_finger_count != previous_count:
    if total_finger_count in sounds_mapping:
      pygame.mixer.music.load(sounds_mapping[total_finger_count])
      pygame.mixer.music.play()
    previous_count = total_finger_count

  # 6. Display current finger count and swara info on screen
  cv2.putText(
      frame,
      f"Fingers: {total_finger_count}",
      (30, 50),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      (0, 0, 255),
      2,
      cv2.LINE_AA,
  )

  # 7. Display live feed
  cv2.imshow("Hand Gesture Swara Player", frame)

  # 8. Exit condition
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
landmarker.close()