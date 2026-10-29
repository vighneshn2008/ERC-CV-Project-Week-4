import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import hand_landmarker

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

print("Milestone 4 running. Press 'q' to quit.")

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

  # 5. Draw the skeleton on each detected hand
  if result.hand_landmarks:
    for hand_landmarks in result.hand_landmarks:
      # Convert normalized fractions (0 to 1) to actual pixel coordinates and draw circles on landmarks
      for lm in hand_landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)  # Red dots for joints

  # 6. Display the live feed
  cv2.imshow("Milestone 4 - Hand Skeleton", frame)

  # 7. Exit condition
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
landmarker.close()