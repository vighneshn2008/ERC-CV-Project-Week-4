import os
import time
import cv2
import pygame
import mediapipe as mp

# --- Configuration ---
MODEL_PATH = "hand_landmarker.task"
SOUNDS_DIR = "sound_swaras"

# --- Initialize Pygame Mixer ---
pygame.mixer.init()

# --- Map Counts to Swaras and Blue-Themed Colors (BGR Format) ---
swara_config = {
    1: ("Sa", os.path.join(SOUNDS_DIR, "1_Sa.mp3"), (200, 100, 50)),  # Steel Blue
    2: ("Re", os.path.join(SOUNDS_DIR, "2_Re.mp3"), (220, 150, 80)),  # Dodger Blue
    3: ("Ga", os.path.join(SOUNDS_DIR, "3_Ga.mp3"), (250, 200, 100)),  # Sky Blue
    4: ("Ma", os.path.join(SOUNDS_DIR, "4_Ma.mp3"), (255, 180, 50)),  # Deep Sky Blue
    5: ("Pa", os.path.join(SOUNDS_DIR, "5_Pa.mp3"), (255, 220, 120)),  # Light Cyan
    6: ("Dha", os.path.join(SOUNDS_DIR, "6_Dha.mp3"), (180, 120, 30)),  # Cobalt Blue
    7: ("Ni", os.path.join(SOUNDS_DIR, "7_Ni.mp3"), (240, 130, 20)),  # Royal Blue
    8: (
        "Upper Sa",
        os.path.join(SOUNDS_DIR, "8_Sa_upper.mp3"),
        (255, 255, 0),
    ),  # Bright Cyan / Aqua
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
previous_count = -1
current_swara_name = "Silence"
current_tint = (100, 50, 20)
is_muted = False
current_volume = 0.8  # Default volume at 70%
pygame.mixer.music.set_volume(current_volume)

print(
    "Hand Gesture Swara Player running. Press '=' to Volume Up, '-' to Volume"
    " Down, 'm' to Mute, 'q' to quit."
)

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
        if thumb_tip_x > thumb_joint_x:
          hand_count += 1
      else:
        if thumb_tip_x < thumb_joint_x:
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
        cv2.circle(frame, (px, py), 5, (255, 200, 0), -1)

  # 5. Audio Trigger & State Logic
  if total_finger_count != previous_count:
    if total_finger_count in swara_config:
      name, path, tint = swara_config[total_finger_count]
      if not is_muted:
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
      current_swara_name = f"Swara: {name}"
      current_tint = tint
    else:
      current_swara_name = "Silence"
      current_tint = (100, 50, 20)
    previous_count = total_finger_count

  # 6. Apply Dynamic Blue Screen Tint
  overlay = frame.copy()
  cv2.rectangle(overlay, (0, 0), (w, h), current_tint, -1)
  alpha = 0.08 if is_muted else 0.15
  cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

  # --- BLUE THEME CUSTOM UI OVERLAYS ---

  # Top Title Box
  cv2.rectangle(frame, (20, 15), (w - 20, 75), (40, 20, 10), -1)
  cv2.rectangle(frame, (20, 15), (w - 20, 75), (255, 200, 0), 2)
  cv2.putText(
      frame,
      "Electronics and Robotics club - BITS Goa",
      (35, 53),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.75,
      (255, 255, 255),
      2,
      cv2.LINE_AA,
  )

  # Middle Info Box (Fingers, Swara, Volume, and Mute Status)
  cv2.rectangle(frame, (20, 85), (w - 20, 150), (40, 20, 10), -1)
  cv2.rectangle(frame, (20, 85), (w - 20, 150), (255, 150, 50), 2)
  cv2.putText(
      frame,
      f"Fingers: {total_finger_count}",
      (35, 128),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      (255, 220, 100),
      2,
      cv2.LINE_AA,
  )
  cv2.putText(
      frame,
      current_swara_name,
      (w - 460, 128),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      (255, 255, 0),
      2,
      cv2.LINE_AA,
  )

  # Display Volume Percentage & Mute Status
  vol_percent = int(current_volume * 100)
  status_text = (
      "MUTED" if is_muted else f"Vol: {vol_percent}% (=/+ , -)"
  )
  status_color = (0, 0, 255) if is_muted else (0, 255, 0)
  cv2.putText(
      frame,
      status_text,
      (w - 300, 128),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.6,
      status_color,
      2,
      cv2.LINE_AA,
  )

  # Bottom Footer Box
  cv2.rectangle(frame, (20, h - 55), (w - 20, h - 15), (40, 20, 10), -1)
  cv2.rectangle(frame, (20, h - 55), (w - 20, h - 15), (150, 100, 50), 1)
  cv2.putText(
      frame,
      "made by Vighnesh Nauso Shetye",
      (35, h - 28),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.65,
      (220, 220, 220),
      1,
      cv2.LINE_AA,
  )

  # 7. Display live feed
  cv2.imshow("Hand Gesture Swara Player - Blue Theme", frame)

  # 8. Keyboard Controls ('q' to quit, 'm' to mute, '='/'+' to vol up, '-' to vol down)
  key = cv2.waitKey(1) & 0xFF
  if key == ord("q"):
    break
  elif key == ord("m"):
    is_muted = not is_muted
    if is_muted:
      pygame.mixer.music.set_volume(0)
      print("Audio Muted")
    else:
      pygame.mixer.music.set_volume(current_volume)
      print(f"Audio Unmuted (Volume: {int(current_volume * 100)}%)")
  elif key == ord("=") or key == ord("+"):
    current_volume = min(1.0, current_volume + 0.1)
    if not is_muted:
      pygame.mixer.music.set_volume(current_volume)
    print(f"Volume Up: {int(current_volume * 100)}%")
  elif key == ord("-"):
    current_volume = max(0.0, current_volume - 0.1)
    if not is_muted:
      pygame.mixer.music.set_volume(current_volume)
    print(f"Volume Down: {int(current_volume * 100)}%")

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
landmarker.close()