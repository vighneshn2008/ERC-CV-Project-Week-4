import cv2

# --- Initialize Webcam ---
# 0 refers to the default built-in webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
  print("Error: Could not open webcam.")
  exit()

print("Milestone 1 running. Press 'q' to quit.")

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    break

  # 1. Mirror the frame horizontally for intuitive interaction
  frame = cv2.flip(frame, 1)

  # 2. Display the live feed
  cv2.imshow("Milestone 1 - Webcam Feed", frame)

  # 3. Wait for 1ms and check if 'q' is pressed to exit
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()