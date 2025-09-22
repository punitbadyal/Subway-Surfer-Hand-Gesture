import cv2
import mediapipe as mp
import pyautogui
import time


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)


cap = cv2.VideoCapture(0)
cap.set(3, 640)  
cap.set(4, 480)  


prev_x, prev_y = 0, 0
gesture_cooldown = 0.4  
last_gesture_time = time.time()


SWIPE_THRESHOLD = 25 
VERTICAL_THRESHOLD = 25  

def send_key(key):
    """Send key press with cooldown."""
    global last_gesture_time
    if time.time() - last_gesture_time > gesture_cooldown:
        pyautogui.press(key)
        print(f"Action: {key}")
        last_gesture_time = time.time()

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            
            index_finger = hand_landmarks.landmark[8]
            x, y = int(index_finger.x * frame.shape[1]), int(index_finger.y * frame.shape[0])

            if prev_x != 0 and prev_y != 0:
                dx = x - prev_x
                dy = y - prev_y

                
                if dx < -SWIPE_THRESHOLD:
                    send_key('left')
            
                elif dx > SWIPE_THRESHOLD:
                    send_key('right')
            
                elif dy < -VERTICAL_THRESHOLD:
                    send_key('up')
                
                elif dy > VERTICAL_THRESHOLD:
                    send_key('down')

            prev_x, prev_y = x, y

    cv2.imshow("Subway Surfers Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
