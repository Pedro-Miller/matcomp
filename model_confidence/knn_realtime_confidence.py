import joblib
import cv2 as cv
import mediapipe as mp
import numpy as np

tol = 0.88

from knn_train_confidence import KNNClassifier 

# Carregar o modelo treinado
knn_model = joblib.load('knn_model_confidence.pkl')

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
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append([lm.x, lm.y, lm.z])
                    
                    # Calcular distâncias relativas entre todos os pares de landmarks
                    relative_distances = []
                    for i in range(len(landmarks)):
                        for j in range(i + 1, len(landmarks)):
                            distance = np.linalg.norm(np.array(landmarks[i]) - np.array(landmarks[j]))
                            relative_distances.append(distance)
                    
                    # Verificar se a quantidade de dados de entrada é compatível com a do modelo
                    if len(relative_distances) == len(self.model.X_train[0]):
                        predictions, confidences = self.model.predict([relative_distances])
                        predicted_letter = predictions[0]
                        confidence = confidences[0]

                        if confidence >= tol:
                            cv.putText(image, f'Prediction: {predicted_letter} ({confidence:.2f})', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
                        else:
                            cv.putText(image, f'Prediction: NADA', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            cv.imshow('MediaPipe Hands', image)
            if cv.waitKey(1) & 0xFF == ord('0'):
                break

        vid.release()
        cv.destroyAllWindows()

# Iniciar a classificação em tempo real
real_time_classifier = RealTimeClassifier(knn_model)
real_time_classifier.captureCamera()
