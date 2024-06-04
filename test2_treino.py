import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Carregar os dados
data = pd.read_csv('landmarks_data.csv')

# Separar as características (landmarks) e o rótulo (letra)
X = data.drop('letter', axis=1)
y = data['letter']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar um modelo de Random Forest
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = clf.predict(X_test)

# Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
joblib.dump(clf, 'random_forest_model.pkl')
print(f'Accuracy: {accuracy}')
