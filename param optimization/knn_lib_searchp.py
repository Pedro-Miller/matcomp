import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Carrega os dados do arquivo CSV
data = pd.read_csv('landmarks_data.csv')

# Separa as coordenadas das landmarks (X) e os rótulos (y)
X = data.drop('letter', axis=1).values
y = data['letter'].values

# Codifica os rótulos de caracteres para inteiros, para os metodos funcionarem
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Divide os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Determina o valor de k
k = 57

# Função de avaliação acurácia média com validação cruzada
def evaluate_p(p):
    knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=p)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    return scores.mean()

# Definiçao dos parametros iniciais

p_min = 1           #borda menor do intervalo de busca
p_max = 501         #borda maior do intervalo de busca
interval_size = 100 #tamanho inicial do intervalo de corte

best_p = None
best_accuracy = 0.0
results = []

# Busca iterativa
while interval_size >= 1:
    print(f"Buscando de {p_min} até {p_max} a cada {interval_size}")
    best_p = None
    best_accuracy = None
    for p in range(p_min, p_max + 1, interval_size):
        current_accuracy = evaluate_p(p)
        results.append({'p': p, 'accuracy': current_accuracy})
        print(f"p = {p}, Acurácia = {current_accuracy}")
        
        if  best_accuracy == None or current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_p = p
    print("Best p = ",best_p)
    # Ajusta o intervalo de busca conforme o valor ótimo de p se aproxima
    if best_p is not None and interval_size != 1:

        # Reduz o intervalo de corte
        interval_size = max(1, interval_size // 10)
        
        # Calcula novos limites do intervalo de busca
        best_pl = best_p
        if best_pl - interval_size* 5 < 1:
            print(best_pl,(interval_size * 5) )
            best_pl = best_p + interval_size * 5
        p_min = max(1, best_pl - interval_size * 10)
        p_max = min(501, best_pl + interval_size * 10)
    else:
        break

print (f"O valor ótimo de p é {best_p}, com acurácia associada de {best_accuracy}.")

# Salva o par com melhor p em um arquivo CSV
par_final = pd.DataFrame((best_p,best_accuracy))
par_final.to_csv('param optimization\best_p_acc.csv', index=False)


# Salva todos os pares em um arquivo CSV
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='p', ascending=True) #ordena em função de p
results_df = results_df.drop_duplicates(subset='p', keep='last') #apaga duplicatas
results_df.to_csv('param optimization\p_vs_accuracy_results.csv', index=False)

