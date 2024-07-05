import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import heapq

from knn_train_confidence import KNNClassifier

def optimize_k(self, k_start = 1, k_end = 3, top_n = 5):
        resultados = []

        for i in range(k_start, k_end):
            knn = KNNClassifier(k=i)
            knn.fit(self.X_train, self.y_train)

            # Faz previsões no conjunto de teste
            y_pred, confidences = knn.predict(self.X_test)

            # Avalia a precisão do modelo
            accuracy = accuracy_score(self.y_test, y_pred)
            resultados.append((i, accuracy))
        maiores_tuplas = heapq.nlargest(10, resultados, key=lambda x: x[1])
        # Imprime os 10 maiores valores e seus índices
        print(resultados)
        for indice, valor in maiores_tuplas:
            print(f"Índice: {indice}, Valor: {valor}")