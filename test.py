import cv2 as cv
import numpy as np
import time
import mediapipe as mp
import pandas as pd

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(min_detection_confidence=0.3)

class Vision:
    def __init__(self):
        self.landmarks_data = []

    def captureCamera(self):
        vid = cv.VideoCapture(0)
        elapsed_time = 0
        t0 = time.time()
        while(elapsed_time < 25):
            success, image = vid.read()
            image = cv.resize(image, (0, 0), fx=1.7, fy=1.7)
            image_height, image_width, _ = image.shape
            image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x_max = 0
                    y_max = 0
                    x_min = image_width
                    y_min = image_height
                    for lm in hand_landmarks.landmark:
                        x, y = int(lm.x * image_width), int(lm.y * image_height)
                        if x > x_max:
                            x_max = x
                        if x < x_min:
                            x_min = x
                        if y > y_max:
                            y_max = y
                        if y < y_min:
                            y_min = y
                        self.landmarks_data.append({
                            'x': lm.x,
                            'y': lm.y,
                            'z': lm.z,
                            'image_width': image_width,
                            'image_height': image_height,
                            'time': elapsed_time
                        })
                    cv.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            cv.imshow('MediaPipe Hands', image)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            elapsed_time = time.time() - t0
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        vid.release()
        cv.destroyAllWindows()
        self.save_landmarks_data()

    def save_landmarks_data(self):
        df = pd.DataFrame(self.landmarks_data)
        df.to_csv('landmarks_data.csv', index=False)
        print('Landmarks data saved to landmarks_data.csv')

vision = Vision()
vision.captureCamera()
