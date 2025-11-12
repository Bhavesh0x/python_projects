import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Get screen size
screen_w, screen_h = pyautogui.size()

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Smooth movement
prev_x, prev_y = 0, 0
smooth = 1  # Lower = faster movement, higher = smoother

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            lm = handLms.landmark
            # Index finger tip (8) and thumb tip (4)
            ix, iy = int(lm[8].x * w), int(lm[8].y * h)
            tx, ty = int(lm[4].x * w), int(lm[4].y * h)

            # Convert hand coords to screen coords
            screen_x = np.interp(ix, (100, w - 100), (0, screen_w))
            screen_y = np.interp(iy, (100, h - 100), (0, screen_h))

            # Smooth cursor motion
            curr_x = prev_x + (screen_x - prev_x) / smooth
            curr_y = prev_y + (screen_y - prev_y) / smooth
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Draw points
            cv2.circle(frame, (ix, iy), 10, (0, 255, 0), -1)
            cv2.circle(frame, (tx, ty), 10, (0, 0, 255), -1)
            cv2.line(frame, (ix, iy), (tx, ty), (255, 255, 0), 2)

            # Check distance for click
            distance = math.hypot(tx - ix, ty - iy)
            if distance < 30:
                pyautogui.click()
                cv2.putText(frame, "Click", (ix, iy - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("AI Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

