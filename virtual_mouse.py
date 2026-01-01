import cv2
import mediapipe as mp
import pyautogui

web = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hand_detector = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

screen_width, screen_height = pyautogui.size()
index_y = 0

while True:
    success, img = web.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_height, img_width, _ = img.shape
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output = hand_detector.process(rgb_img)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
            landmarks = hand.landmark

            for id, lm in enumerate(landmarks):
                x = int(lm.x * img_width)
                y = int(lm.y * img_height)

                if id == 8:  # Index finger
                    cv2.circle(img, (x, y), 20, (0, 255, 255), -1)
                    index_x = screen_width / img_width * x
                    index_y = screen_height / img_height * y
                    pyautogui.moveTo(index_x, index_y)

                if id == 4:  # Thumb
                    cv2.circle(img, (x, y), 20, (0, 255, 255), -1)
                    thumb_y = screen_height / img_height * y

                    if abs(index_y - thumb_y) < 20:
                        pyautogui.click()
                        pyautogui.sleep(1)

    cv2.imshow("Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

web.release()
cv2.destroyAllWindows()
