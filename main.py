import time
import cv2
import mediapipe as mp

# --- Configuration ---
# Replace with the actual path to your downloaded hand_landmarker.task file
MODEL_PATH = "hand_landmarker.task"

# --- Initialize MediaPipe Hand Landmarker (Milestone 2) ---
# The pattern: where is the model -> how should it run -> build it
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

print("Milestone 2 running. Press 'q' to quit.")

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    break

  # 1. Mirror the frame horizontally
  frame = cv2.flip(frame, 1)

  # 2. Display the live feed
  cv2.imshow("Milestone 2 - Detector Setup", frame)

  # 3. Exit condition
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
landmarker.close()  # Always close the landmarker model at the end!