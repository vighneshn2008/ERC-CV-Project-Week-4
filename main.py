import time
import cv2
import mediapipe as mp

# --- Configuration ---
MODEL_PATH = "hand_landmarker.task"

# --- Initialize MediaPipe Hand Landmarker ---
base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,  # Tracks up to two hands to allow counts up to 10
)
landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

# --- Initialize Webcam ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
  print("Error: Could not open webcam.")
  exit()

start_time = time.time()

print("Milestone 5 running. Press 'q' to quit.")

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    break

  # 1. Mirror the frame horizontally
  frame = cv2.flip(frame, 1)
  h, w, _ = frame.shape

  # 2. Prepare frame for MediaPipe (BGR -> RGB)
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

  # 3. Compute timestamp in milliseconds
  timestamp_ms = int((time.time() - start_time) * 1000)

  # 4. Detect hands
  result = landmarker.detect_for_video(mp_image, timestamp_ms)

  total_finger_count = 0

  if result.hand_landmarks:
    for i, hand_landmarks in enumerate(result.hand_landmarks):
      # Get handedness ("Left" or "Right")
      label = result.handedness[i][0].category_name

      hand_count = 0

      # --- Thumb Logic ---
      # Compare x-coordinate of tip (4) and IP joint (3).
      # Direction depends on whether it's a left or right hand (mirrored feed check).
      thumb_tip_x = hand_landmarks[4].x
      thumb_joint_x = hand_landmarks[3].x

      if label == "Right":
        if thumb_tip_x < thumb_joint_x:
          hand_count += 1
      else:  # Left hand
        if thumb_tip_x > thumb_joint_x:
          hand_count += 1

      # --- Four Fingers Logic (Index, Middle, Ring, Pinky) ---
      # Tips: 8, 12, 16, 20 | Lower joints (PIP): 6, 10, 14, 18
      finger_tips = [8, 12, 16, 20]
      finger_pips = [6, 10, 14, 18]

      for tip, pip in zip(finger_tips, finger_pips):
        # y-axis points down, so a smaller y value means physically higher
        if hand_landmarks[tip].y < hand_landmarks[pip].y:
          hand_count += 1

      total_finger_count += hand_count

      # Draw hand landmarks on the display frame
      for lm in hand_landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

  # 5. Display the total finger count on screen
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

  # 6. Display live feed
  cv2.imshow("Milestone 5 - Finger Counting", frame)

  # 7. Exit condition
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
landmarker.close()