import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

class KNNClassifier:

    def __init__(self, k=5):
        self.k = k
        self.label_encoder = LabelEncoder()

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = self.label_encoder.fit_transform(y_train)

    def predict(self, X_test):
        predictions = []
        confidences = []

        for sample in X_test:
            distances = [np.linalg.norm(sample - x) for x in self.X_train]
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_indices]

            # Determina a classe predita com base na contagem de ocorrências
            counts = np.bincount(nearest_labels)
            prediction = np.argmax(counts)
            predictions.append(self.label_encoder.inverse_transform([prediction])[0])

            # Calcula a confiança como a proporção de vizinhos que são da classe predita
            confidence = counts[prediction] / self.k
            confidences.append(confidence)

        return predictions, confidences

# Carrega os dados do arquivo CSV
data = pd.read_csv('landmarks_data.csv')

# Separa as coordenadas das landmarks (X) e os rótulos (y)
X = data.drop('letter', axis=1).values
y = data['letter'].values

# Divide os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializa e treina o modelo KNN
knn = KNNClassifier(k=125)
knn.fit(X_train, y_train)

# Faz previsões no conjunto de teste
y_pred, confidences = knn.predict(X_test)

# Avalia a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Salva o modelo treinado
joblib.dump(knn, 'knn_model_confidence.pkl')
