import cv2 as cv
import mediapipe as mp
import pandas as pd
import time
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(min_detection_confidence=0.3)

class DataCollector:
    def __init__(self):
        self.landmarks_data = []

    def captureCamera(self):
        vid = cv.VideoCapture(0)
        while True:
            current_letter = input("Enter the letter to capture data for (or 'q' to quit): ")
            if current_letter.lower() == 'q':
                break

            elapsed_time = 0
            t0 = time.time()
            print(f"Capturing data for letter: {current_letter}")

            while elapsed_time < 25:
                success, image = vid.read()
                if not success:
                    print("Failed to capture image")
                    break
                
                image = cv.resize(image, (0, 0), fx=1.7, fy=1.7)
                image_height, image_width, _ = image.shape
                image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
                results = hands.process(image)
                image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks_row = [current_letter]
                        for lm in hand_landmarks.landmark:
                            landmarks_row.extend([lm.x, lm.y, lm.z])
                        self.landmarks_data.append(landmarks_row)
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

            print(f"Finished capturing data for letter: {current_letter}")
            self.save_landmarks_data()
            self.landmarks_data = []  # Clear data after saving to avoid memory issues

        vid.release()
        cv.destroyAllWindows()

    def save_landmarks_data(self):
        columns = ['letter'] + [f'{coord}_{i}' for i in range(21) for coord in ['x', 'y', 'z']]
        df = pd.DataFrame(self.landmarks_data, columns=columns)
        
        if os.path.exists('landmarks_data.csv'):
            df.to_csv('landmarks_data.csv', mode='a', header=False, index=False)
        else:
            df.to_csv('landmarks_data.csv', index=False)
        
        print('Landmarks data saved to landmarks_data.csv')

collector = DataCollector()
collector.captureCamera()