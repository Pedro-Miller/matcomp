import mss
from PIL import Image, ImageGrab

import cv2 as cv
import numpy as np
import time
import mediapipe as mp
import pandas as pd

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(min_detection_confidence = 0.3)

class Vision:
    def __init__(self):
        pass

    def captureScreen(self, widht, height):
        img = None
        t0 = time.time()
        n_frames = 1
        monitor = {"top": 250, "left": 1100, "width": widht, "height": height}
        elapsed_time = 0
        with mss.mss() as sct:
            while elapsed_time < 25:
                img = sct.grab(monitor)
                img = np.array(img)                         # Convert to NumPy array
                img = cv.cvtColor(img, cv.COLOR_RGB2BGR)  # Convert RGB to BGR color
                
                results = hands.process(img)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

                small = cv.resize(img, (0, 0), fx=2, fy=2)
                small = cv.cvtColor(small, cv.COLOR_BGR2RGB)
                cv.imshow("Hand Detection", small)

                # Break loop and end test
                key = cv.waitKey(1)
                if key == ord('q'):
                    break
                
                elapsed_time = time.time() - t0
                avg_fps = (n_frames / elapsed_time)
                print("Average FPS: " + str(avg_fps))
                n_frames += 1
    def captureCamera(self):
        vid = cv.VideoCapture(0)
        elapsed_time = 0
        t0 = time.time()
        while(elapsed_time < 25): 
            success, image = vid.read()
            image = cv.resize(image, (0, 0), fx=1.7, fy=1.7) 
            image_height, image_width, _ = image.shape
            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            results = hands.process(image)
            image = cv.cvtColor(image,cv.COLOR_RGB2BGR)
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
                    cv.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    for ids, landmrk in enumerate(hand_landmarks.landmark):
                        # print(ids, landmrk)
                        cx, cy = landmrk.x * image_width, landmrk.y*image_height
                        print(cx, cy)                                                               
                    mp_drawing.draw_landmarks(                                                 
                    image,
                    hand_landmarks,mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            cv.imshow('MediaPipe Hands', image)
            if cv.waitKey(1) & 0xFF == ord('q'): 
                        break
            cv.waitKey(1)

            elapsed_time = time.time() - t0
            # the 'q' button is set as the 
            # quitting button you may use any 
            # desired button of your choice 
            if cv.waitKey(1) & 0xFF == ord('q'): 
                break
  

vision = Vision()
vision.captureCamera()
#vision.captureScreen(800,600)




    