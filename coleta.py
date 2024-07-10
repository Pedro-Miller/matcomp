import cv2 as cv  # Biblioteca para manipulação de imagens
import mediapipe as mp  # Biblioteca para detecção de mãos
import pandas as pd  # Biblioteca para manipulação de dados
import time  # Biblioteca para medição de tempo
import os  # Biblioteca para manipulação de arquivos
import numpy as np  # Biblioteca para cálculos numéricos

# Configuração do MediaPipe Hands
mp_hands = mp.solutions.hands  # Módulo de soluções de mãos do MediaPipe
mp_drawing = mp.solutions.drawing_utils  # Utilitários de desenho do MediaPipe
mp_drawing_styles = mp.solutions.drawing_styles  # Estilos de desenho do MediaPipe

# Inicialização do detector de mãos com confiança mínima de detecção
hands = mp_hands.Hands(min_detection_confidence=0.3)

#Classe para coletar dados de landmarks de mãos usando a webcam e MediaPipe Hands.
class DataCollector:

    def __init__(self):

        self.landmarks_data = []  # Lista para armazenar os dados dos landmarks

    def captureCamera(self):
    
        #Captura imagens da webcam, processa os landmarks das mãos e salva os dados coletados.

        vid = cv.VideoCapture(0)  # Inicializa a captura de vídeo
        while True:
            current_letter = input("Enter the letter to capture data for (or '0' to quit): ").upper()  # Solicita ao usuário a letra para capturar dados
            if current_letter == '0':  # Condição para sair do loop
                break

            elapsed_time = 0  # Tempo decorrido
            t0 = time.time()  # Tempo inicial
            print(f"Capturing data for letter: {current_letter}")  # Informa a letra sendo capturada

            while elapsed_time < 8:  # Captura dados por 8 segundos
                success, image = vid.read()  # Captura uma imagem da webcam
                if not success:  # Verifica se a captura foi bem-sucedida
                    print("Failed to capture image")
                    break
                
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
                        
                        # Calcula as distâncias relativas entre todos os pares de landmarks
                        relative_distances = []
                        for i in range(len(landmarks)):
                            for j in range(i + 1, len(landmarks)):
                                distance = np.linalg.norm(np.array(landmarks[i]) - np.array(landmarks[j]))  # Calcula a distância
                                relative_distances.append(distance)  # Adiciona a distância à lista
                        
                        self.landmarks_data.append([current_letter] + relative_distances)  # Adiciona a letra e as distâncias aos dados
                        
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

                elapsed_time = time.time() - t0  # Atualiza o tempo decorrido

            print(f"Finished capturing data for letter: {current_letter}")  # Informa que a captura terminou
            self.save_landmarks_data()  # Salva os dados dos landmarks
            self.landmarks_data = []  # Limpa os dados após salvar para evitar problemas de memória

        vid.release()  # Libera a captura de vídeo
        cv.destroyAllWindows()  # Fecha todas as janelas

    def save_landmarks_data(self):
        """
        Salva os dados de landmarks coletados em um arquivo CSV.
        """
        num_landmarks = 21  # Número de landmarks
        num_distances = num_landmarks * (num_landmarks - 1) // 2  # Número de distâncias relativas
        columns = ['letter'] + [f'dist_{i}' for i in range(num_distances)]  # Nomes das colunas do CSV
        df = pd.DataFrame(self.landmarks_data, columns=columns)  # Cria um DataFrame com os dados
        
        # Salva os dados no arquivo CSV
        if os.path.exists('landmarks_data.csv'):  # Verifica se o arquivo já existe
            df.to_csv('landmarks_data.csv', mode='a', header=False, index=False)  # Adiciona os dados ao arquivo existente
        else:
            df.to_csv('landmarks_data.csv', index=False)  # Cria um novo arquivo com os dados
        
        print('Landmarks data saved to landmarks_data.csv')  # Informa que os dados foram salvos

# Instancia e executa o coletor de dados
collector = DataCollector()
collector.captureCamera()
