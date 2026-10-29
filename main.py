import time
import cv2
import mediapipe as mp

# --- Configuration ---
MODEL_PATH = "hand_landmarker.task"

# --- Initialize MediaPipe Hand Landmarker (Milestone 2) ---
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

# Record the start time for computing video timestamps
start_time = time.time()

print("Milestone 3 running. Press 'q' to quit.")

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    break

  # 1. Mirror the frame horizontally
  frame = cv2.flip(frame, 1)

  # 2. Prepare frame for MediaPipe (BGR -> RGB)
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

  # 3. Compute timestamp in milliseconds since start
  timestamp_ms = int((time.time() - start_time) * 1000)

  # 4. Detect hands in the current video frame
  result = landmarker.detect_for_video(mp_image, timestamp_ms)

  # Checkpoint: Print the number of detected hands each frame
  print(f"Hands detected: {len(result.hand_landmarks)}")

  # 5. Display the live feed
  cv2.imshow("Milestone 3 - Detection", frame)

  # 6. Exit condition
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
landmarker.close()