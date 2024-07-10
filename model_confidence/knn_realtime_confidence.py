import joblib  # Biblioteca para salvar e carregar modelos
import cv2 as cv  # Biblioteca para manipulação de imagens
import mediapipe as mp  # Biblioteca para detecção de mãos
import numpy as np  # Biblioteca para cálculos numéricos

tol = 0.88  # Tolerância mínima para exibir a previsão

from knn_train_confidence import KNNClassifier  # Importa a classe KNNClassifier

# Carregar o modelo treinado
knn_model = joblib.load('knn_model_confidence.pkl')  # Carrega o modelo KNN treinado

# Configuração do MediaPipe Hands
mp_hands = mp.solutions.hands  # Módulo de soluções de mãos do MediaPipe
mp_drawing = mp.solutions.drawing_utils  # Utilitários de desenho do MediaPipe
mp_drawing_styles = mp.solutions.drawing_styles  # Estilos de desenho do MediaPipe

# Inicializa o detector de mãos com confiança mínima de detecção
hands = mp_hands.Hands(min_detection_confidence=0.3)

class RealTimeClassifier:
    # Classe para classificação em tempo real

    def __init__(self, model):
        # Inicializa o classificador em tempo real
        # Parâmetros:
        # model: Modelo treinado a ser usado para a classificação
        self.model = model  # Armazena o modelo

    def captureCamera(self):
        # Captura imagens da webcam e faz a classificação em tempo real
        vid = cv.VideoCapture(0)  # Inicializa a captura de vídeo
        while True:
            success, image = vid.read()  # Captura uma imagem da webcam
            image = cv.resize(image, (0, 0), fx=1.7, fy=1.7)  # Redimensiona a imagem
            image_height, image_width, _ = image.shape  # Obtém as dimensões da imagem
            image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)  # Converte a imagem para RGB e inverte horizontalmente
            results = hands.process(image)  # Processa a imagem para detectar mãos
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)  # Converte a imagem de volta para BGR
            if results.multi_hand_landmarks:  # Verifica se foram detectadas mãos
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []  # Lista para armazenar os landmarks
                    for lm in hand_landmarks.landmark:  # Itera sobre os landmarks detectados
                        landmarks.append([lm.x, lm.y, lm.z])  # Adiciona os landmarks à lista
                    
                    # Calcular distâncias relativas entre todos os pares de landmarks
                    relative_distances = []  # Lista para armazenar as distâncias
                    for i in range(len(landmarks)):
                        for j in range(i + 1, len(landmarks)):
                            distance = np.linalg.norm(np.array(landmarks[i]) - np.array(landmarks[j]))  # Calcula a distância
                            relative_distances.append(distance)  # Adiciona a distância à lista
                    
                    # Verificar se a quantidade de dados de entrada é compatível com a do modelo
                    if len(relative_distances) == len(self.model.X_train[0]):  # Verifica compatibilidade dos dados
                        predictions, confidences = self.model.predict([relative_distances])  # Faz previsões
                        predicted_letter = predictions[0]  # Obtém a letra predita
                        confidence = confidences[0]  # Obtém a confiança

                        if confidence >= tol:  # Verifica se a confiança é maior ou igual à tolerância
                            cv.putText(image, f'Prediction: {predicted_letter} ({confidence:.2f})', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
                        else:
                            cv.putText(image, 'Prediction: NADA', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

                    # Desenha os landmarks e as conexões na imagem
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            cv.imshow('MediaPipe Hands', image)  # Mostra a imagem processada
            if cv.waitKey(1) & 0xFF == ord('0'):  # Verifica se a tecla '0' foi pressionada para sair do loop
                break

        vid.release()  # Libera a captura de vídeo
        cv.destroyAllWindows()  # Fecha todas as janelas

# Iniciar a classificação em tempo real
real_time_classifier = RealTimeClassifier(knn_model)  # Instancia o classificador em tempo real
real_time_classifier.captureCamera()  # Inicia a captura da câmera e a classificação
