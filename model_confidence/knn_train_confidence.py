import numpy as np  # Biblioteca para cálculos numéricos
import pandas as pd  # Biblioteca para manipulação de dados
from sklearn.model_selection import train_test_split  # Função para dividir os dados em conjuntos de treino e teste
from sklearn.metrics import accuracy_score  # Função para calcular a precisão do modelo
from sklearn.preprocessing import LabelEncoder  # Codificador para transformar rótulos em números
import joblib  # Biblioteca para salvar e carregar modelos

class KNNClassifier:
    # Classe para o algoritmo K-Nearest Neighbors (KNN) com cálculo de confiança

    def __init__(self, k=5):
        # Inicializa o classificador KNN
        # Parâmetros:
        # k (int): Número de vizinhos mais próximos a serem considerados
        self.k = k  # Número de vizinhos
        self.label_encoder = LabelEncoder()  # Inicializa o codificador de rótulos

    def fit(self, X_train, y_train):
        # Treina o classificador KNN
        # Parâmetros:
        # X_train (ndarray): Dados de treinamento
        # y_train (ndarray): Rótulos de treinamento
        self.X_train = X_train  # Armazena os dados de treinamento
        self.y_train = self.label_encoder.fit_transform(y_train)  # Codifica os rótulos de treinamento

    def predict(self, X_test):
        # Faz previsões para os dados de teste
        # Parâmetros:
        # X_test (ndarray): Dados de teste
        # Retorna:
        # predictions (list): Lista de previsões
        # confidences (list): Lista de confianças associadas a cada previsão
        predictions = []  # Lista para armazenar previsões
        confidences = []  # Lista para armazenar confianças

        for sample in X_test:  # Itera sobre cada amostra de teste
            distances = [np.linalg.norm(sample - x) for x in self.X_train]  # Calcula distâncias entre a amostra de teste e todos os dados de treinamento
            nearest_indices = np.argsort(distances)[:self.k]  # Obtém os índices dos k vizinhos mais próximos
            nearest_labels = self.y_train[nearest_indices]  # Obtém os rótulos dos vizinhos mais próximos

            # Determina a classe predita com base na contagem de ocorrências
            counts = np.bincount(nearest_labels)  # Conta a ocorrência de cada rótulo entre os vizinhos
            prediction = np.argmax(counts)  # Obtém o rótulo mais comum
            predictions.append(self.label_encoder.inverse_transform([prediction])[0])  # Adiciona a previsão à lista

            # Calcula a confiança como a proporção de vizinhos que são da classe predita
            confidence = counts[prediction] / self.k  # Calcula a confiança
            confidences.append(confidence)  # Adiciona a confiança à lista

        return predictions, confidences  # Retorna previsões e confianças

# Carrega os dados do arquivo CSV
data = pd.read_csv('landmarks_data.csv')  # Lê os dados do arquivo CSV

# Separa as coordenadas das landmarks (X) e os rótulos (y)
X = data.drop('letter', axis=1).values  # Obtém as coordenadas das landmarks
y = data['letter'].values  # Obtém os rótulos

# Divide os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Divide os dados

# Inicializa e treina o modelo KNN
knn = KNNClassifier(k=15)  # Inicializa o classificador KNN com k=15
knn.fit(X_train, y_train)  # Treina o classificador

# Faz previsões no conjunto de teste
y_pred, confidences = knn.predict(X_test)  # Faz previsões

# Avalia a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)  # Calcula a precisão do modelo
print(f'Accuracy: {accuracy}')  # Exibe a precisão

# Salva o modelo treinado
joblib.dump(knn, 'knn_model_confidence.pkl')  # Salva o modelo em um arquivo
