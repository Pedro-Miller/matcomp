import joblib
import cv2 as cv
import mediapipe as mp
import numpy as np

# Carregar o modelo treinado
clf = joblib.load('random_forest_model.pkl')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(min_detection_confidence=0.3)

class RealTimeClassifier:
    def __init__(self, model):
        self.model = model

    def captureCamera(self):
        vid = cv.VideoCapture(0)
        while True:
            success, image = vid.read()
            image = cv.resize(image, (0, 0), fx=1.7, fy=1.7)
            image_height, image_width, _ = image.shape
            image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks_row = []
                    for lm in hand_landmarks.landmark:
                        landmarks_row.extend([lm.x, lm.y, lm.z])
                    prediction = self.model.predict([landmarks_row])
                    cv.putText(image, f'Prediction: {prediction[0]}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            cv.imshow('MediaPipe Hands', image)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        vid.release()
        cv.destroyAllWindows()

# Salvar o modelo treinado
joblib.dump(clf, 'random_forest_model.pkl')

# Carregar o modelo e iniciar a classificação em tempo real
real_time_classifier = RealTimeClassifier(clf)
real_time_classifier.captureCamera()
