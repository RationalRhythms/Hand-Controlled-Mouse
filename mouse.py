import cv2
import mediapipe as mp
import numpy as np
import autopy
import math
import time

class HandMouseController:
    def __init__(self, smoothing=0.2, cooldown=0.8):
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, model_complexity=1)
        self.draw = mp.solutions.drawing_utils
        self.screen_w, self.screen_h = autopy.screen.size()

        self.smoothing = smoothing
        self.prev_x = self.screen_w / 2
        self.prev_y = self.screen_h / 2
        self.last_click_time = 0
        self.click_cooldown = cooldown

    def calculate_distance(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def get_thumb_state(self, lm):
        return self.calculate_distance(lm[4], lm[5]) > 0.12

    def get_finger_state(self, lm, tip, pip):
        return lm[tip].y < lm[pip].y

    def run(self):
        while True:
            success, img = self.cap.read()
            if not success:
                continue

            img = cv2.flip(img, 1)
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                lm = hand.landmark

                thumb_up = self.get_thumb_state(lm)
                index_up = self.get_finger_state(lm, 8, 6)
                middle_up = self.get_finger_state(lm, 12, 10)
                ring_up = self.get_finger_state(lm, 16, 14)
                pinky_up = self.get_finger_state(lm, 20, 18)

                ix, iy = int(lm[8].x * w), int(lm[8].y * h)
                screen_x = np.interp(ix, [0, w ], [0, self.screen_w])
                screen_y = np.interp(iy, [0, h ], [0, self.screen_h])

                if index_up and not (middle_up or ring_up or pinky_up):
                    dx = screen_x - self.prev_x
                    dy = screen_y - self.prev_y

                    if abs(dx) > 0.5 or abs(dy) > 0.5:  # Movement threshold
                        smooth_x = self.prev_x + self.smoothing * (screen_x - self.prev_x)
                        smooth_y = self.prev_y + self.smoothing * (screen_y - self.prev_y)
                        autopy.mouse.move(smooth_x, smooth_y)
                        self.prev_x, self.prev_y = smooth_x, smooth_y

                current_time = time.time()
                if index_up and middle_up and not thumb_up:
                    if current_time - self.last_click_time > self.click_cooldown:
                        autopy.mouse.click(autopy.mouse.Button.LEFT)
                        self.last_click_time = current_time
                        cv2.putText(img, "LEFT CLICK", (w // 2 - 50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if index_up and thumb_up and not middle_up:
                    if current_time - self.last_click_time > self.click_cooldown:
                        autopy.mouse.click(autopy.mouse.Button.RIGHT)
                        self.last_click_time = current_time
                        cv2.putText(img, "RIGHT CLICK", (w // 2 - 50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                self.draw.draw_landmarks(img, hand, self.mp_hands.HAND_CONNECTIONS)

                finger_states = [
                    f"Thumb: {'Up' if thumb_up else 'Down'}",
                    f"Index: {'Up' if index_up else 'Down'}",
                    f"Middle: {'Up' if middle_up else 'Down'}"
                ]
                for i, state in enumerate(finger_states):
                    cv2.putText(img, state, (10, 30 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Hand Mouse (autopy)", img)
            if cv2.waitKey(1) == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = HandMouseController()
    app.run()
