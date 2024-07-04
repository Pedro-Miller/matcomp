import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

  # Declarando a classe do classificador KNN (K-Nearest Neighbors)
class KNNClassifier:

    def __init__(self, k=5):
        self.k = k

    def fit(self, X_train, y_train):
        # Recebe os dados (features) (Vetor de posição (x, y, z), como o array X_train e a classe (rótulo) (letra) como o array Y_train)
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        # Os dados de x (posição) serão usados para predizer a qual classe y (letra) cada amostra deve pertencer)
        predictions = []

        # Itera os valores em x
        for sample in X_test:
            # Calcula a distância euclidiana entre o sample e todos os pontos de treinamento
            distances = [np.linalg.norm(sample - x) for x in self.X_train]
            # Obtém os índices dos k vizinhos mais próximos
            nearest_indices = np.argsort(distances)[:self.k]
            # Obtém os rótulos dos k vizinhos mais próximos
            nearest_labels = [self.y_train[i] for i in nearest_indices]
            # Determina o rótulo mais comum entre os vizinhos (vote)
            prediction = max(set(nearest_labels), key=nearest_labels.count)
            #efetua a predição baseado na letra mais votada
            predictions.append(prediction)
        return predictions

# Carrega os dados do arquivo CSV
data = pd.read_csv('landmarks_data.csv')

# Separa as coordenadas das landmarks (X) e os rótulos (y)
X = data.drop('letter', axis=1).values
y = data['letter'].values

# Divide os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializa e treina o modelo KNN
knn = KNNClassifier(k=5)
knn.fit(X_train, y_train)

# Faz previsões no conjunto de teste
y_pred = knn.predict(X_test)

# Avalia a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Salva o modelo treinado
joblib.dump(knn, 'knn_model.pkl')
