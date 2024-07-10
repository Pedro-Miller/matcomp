import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados do arquivo CSV
results_df = pd.read_csv('p_vs_accuracy_results.csv')

par = pd.read_csv('')


best_accuracy = 0.9722429834282622
best_p =1

# Ordenar e remover duplicatas
results_df = results_df.sort_values(by='p', ascending=True)
results_df = results_df.drop_duplicates(subset='p', keep='last')

def plot():
    # Plotar o gráfico de p versus acurácia média
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['p'], results_df['accuracy'], marker='o')
    plt.title('Accuracy vs. p values')
    plt.xlabel('p value')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # Adicionar marcador vertical e linha horizontal para o valor ótimo de p
    plt.axvline(x=best_p, color='r', linestyle='--', label=f'Best p = {best_p}')
    plt.axhline(y=best_accuracy, color='g', linestyle='--', label=f'Best Accuracy = {best_accuracy:.2f}')

    for p in results_df['p']:
        if p != best_p:
            plt.axvline(x=p, color='b', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.legend()



plot()#plot normal
plt.savefig('p_vs_accuracy_plot.png')
plt.show()


plot()#plot com foco
plt.xlim(max(0,best_p-50),best_p+50)
plt.ylim(best_accuracy-0.025,best_accuracy+0.025)

plt.savefig('p_vs_accuracy_plot_focus.png')
plt.show()

