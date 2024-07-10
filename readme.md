# Classificador de Letras em LIBRAS

**`Trabalho final da matéria de Matemática Computacional (COC351 2024.1) do curso de Engenharia de Controle e Automação da UFRJ`**

---
<h1 align='center' >Introdução</h1>
<p>Esse repositório contém o Classificador KNN capaz de identificar Letras em LIBRAS através de visão computacional e regressão em tempo real. Neste trabalho da disciplina de Matemática Computacional, optamos pela abordagem de desenvolvimento dos conceitos de modelos de classificação e algoritmos de busca iterativa. A ideia inicial consiste no desenvolvimento de um script capaz de ler em tempo real através de uma câmera o formato e posição das mãos de uma pessoa e associá-los em tempo real a um caractere do alfabeto em LIBRAS, podendo servir como um interpretador de linguagem de sinais para texto. </p>

# Índice
   1. [Pré-Requisitos](#Pré-Requisitos)
   2. [Como Funciona?](#Como%20Funciona?)
   3. [Como Utilizar?](#Como-utilizar?)
   
# Pré-Requisitos
<p>Foram necessários o uso ou desenvolvimento de algumas ferramentas para cada etapa desse processo, sendo esses: <br>
<b><em>Python3</em></b> , para fundamentar o script <br>
<b><em>OpenCV</em></b>, a biblioteca com modelos para leitura e reconhecimento de imagem <br>
<b><em>MediaPipe</em></b>, a biblioteca com modelo para mapeamento dos gestos <br>
<b><em>Scikit-learn</em></b>, biblioteca com funções para avaliação de modelos de ML <br>
<b><em>Pandas</em></b>, a biblioteca de DataFrames <br>
<b><em>MatplotLib</em></b>, a biblioteca de plotagem gráfica <br>
</p>

Primeiramente instale o Python3 do site oficial:
<a href="https://www.python.org/">Python.org</a>

##### Ou em um ambiente Linux com apt:
    apt install python3

##### Então utilize o instalador de pacotes do Python para baixar as bibliotecas
    pip install opencv-python mediapipe scikit-learn pandas matplotlib
<br>

# Como Funciona?
O KNN (K-Nearest Neighbors) é um algoritmo de aprendizado de máquina supervisionado usado para classificação e regressão. Nesse caso utilizamos esse algoritmo para realizar uma predição baseada em um conjunto de dados.
Funcionamento do KNN
Definição do Parâmetro K: O primeiro passo no uso do KNN é escolher o valor de K, que representa o número de vizinhos mais próximos a serem considerados no processo de classificação. O valor de K influencia diretamente o desempenho do modelo. Um K muito pequeno pode tornar o modelo sensível ao ruído dos dados, enquanto um K muito grande pode suavizar demais as fronteiras entre as classes.
<br><br>

### Cálculo das Distâncias: 
>Dado um novo ponto de dados para ser classificado, o KNN calcula a distância entre este ponto e todos os pontos do conjunto de treinamento. As distâncias mais comumente usadas são a Euclidiana, Manhattan e Minkowski, embora outras métricas de distância possam ser aplicadas dependendo da natureza dos dados.

### Identificação dos Vizinhos Mais Próximos: 
>Após calcular as distâncias, o KNN identifica os K pontos de treinamento mais próximos ao ponto de teste. Esses pontos são chamados de vizinhos.

### Classificação:
>O novo ponto de dados é então classificado com base nas classes dos seus K vizinhos mais próximos. A classe mais comum entre esses vizinhos é atribuída ao novo ponto. Para problemas de regressão, o valor atribuído ao novo ponto é a média dos valores dos K vizinhos mais próximos.

### Vantagens e Desvantagens
><b><em>Vantagens</em></b>:
Simplicidade: O KNN é fácil de entender e implementar.
Versatilidade: Pode ser usado tanto para classificação quanto para regressão.
Sem Suposições de Distribuição: Não faz suposições sobre a distribuição dos dados.<br>
<b><em>Desvantagens</em></b>:
Custo Computacional: Pode ser computacionalmente caro, especialmente com grandes conjuntos de dados, já que envolve o cálculo da distância de todos os pontos de treinamento para cada nova amostra.
Necessidade de Normalização: Os dados precisam ser normalizados, pois o KNN é sensível às escalas das variáveis.
Escolha do Valor de K: A escolha do valor de K pode ser subjetiva e requer experimentação para encontrar o valor ideal.

### Otimização de Parâmetros por Algoritmo de Busca Iterativa
>O modelo KNN possui hiperparâmetros para refinar em função do contexto em que é implementado. Optamos por experimentar com dois hiperparâmetros para obter uma maior acurácia, sendo eles: <b>k</b>, ou número de vizinhos mais próximos a serem avaliados no chute, e <b>p</b>, o expoente Minkowski, que define o tipo de métrica utilizada ao calcular as distâncias.

><b>P</b> é um parâmetro essencial para otimizar o modelo. Ele define o expoente utilizado na fórmula de distância Minkowski, como descrito abaixo:

$$D(x, y) = \left( \sum^n_{i=1} |x_i-y_i|^p \right)^{\frac{1}{p}}$$

>Essa fórmula generaliza as métricas possíveis do cálculo de uma distância entre dois pontos. Por exemplo, <b>p</b>=2 retorna a distância euclidiana, a linha reta entre dois pontos. Ou então <b>p</b>=1 retorna a distância manhattan, o equivalente a soma das componentes ortogonais do vetor entre esses pontos.<br>
>Em função do contexto e propósito do modelo, existe um <b>p</b> que pode trazer melhor acurácia para o modelo, não compondo necessariamente uma métrica euclidiana. A partir de pesquisa e alguns testes, desenvolvemos um algoritmo de natureza heurística para buscar esse <b>p</b> iterativamente. <br>
Observamos inicialmente que a acurácia em função de <b>p</b> era baixa em valores mais elevados, tipo <b>p</b>=501, mas que subia na faixa inicial, a partir de <b>p</b>=1. Formamos um intervalo de busca entre esses dois valores.

#### O algoritmo funciona da seguinte forma:
>O intervalo de busca é seccionado a cada valor de corte <b>n</b>, no nosso caso a cada 100 <br>
Um loop faz treinar o modelo para cada valor de corte (1, 101 … 501) por meio de validação cruzada (5 dobras), e calcula a acurácia média para cada iteração. <br>
Entre esses valores, o com maior acurácia é escolhido como melhor <b>p</b> temporário (best_p), onde o intervalo de busca é reduzido até melhor <b>p</b> mais ou menos o valor de corte. ( max_p, min_p = best_p ± (n) ) <br>
Caso o limite inferior seja menor que 1, o centro da janela é reajustada, somando n, tornando o novo limite inferior igual a 1 (transladando positivamente a janela até que p_min = 1) <br>
O valor de corte (n) é dividido por um valor arbitrário, no nosso caso 10. <br>
O loop retreina novos valores de <b>p</b> entre o intervalo reduzido, separados pelo novo valor de corte, retomando o primeiro passo e afunilando o intervalo de busca a cada iteração. <br>
O loop para quando o valor de corte é inferior a um, impedindo que o último <b>p</b> seja um racional, retornando assim o valor ótimo de <b>p</b>. <br>

Ao longo desse looping, os pares <b>p</b> e acurácia são armazenados numa lista e em seguida num arquivo csv, para estudo e modelagem de gráficos.

Foi observado nesses gráficos que as acurácias se agrupam em plateaux. Mediante experimentação, deduzimos que provavelmente isso se dá ao método que utilizamos para o cálculo da acurácia, que talvez em pequenos intervalos ou maior complexidade associada aos valores de entrada, ele retorna valores em salto, com precisão limitada. Isso é visível  no gráfico a seguir, com valores e datasets ainda experimentais:


---


