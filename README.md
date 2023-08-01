# Previsão de Score de Crédito
Esta documentação tem como objetivo explicar todos os passos necessários para entender e executar a aplicação que realiza a previsão do score de crédito de clientes usando algoritmos de Machine Learning.

# Pré-requisitos
Antes de executar a aplicação, certifique-se de ter o Python instalado em seu sistema e os seguintes pacotes Python instalados:

* pandas
* scikit-learn
Você pode instalar os pacotes usando o gerenciador de pacotes **pip**. Execute o seguinte comando em seu terminal ou prompt de comando:
```
pip install pandas scikit-learn
```
### Passo 1: Importando Bibliotecas
O primeiro passo é importar as bibliotecas necessárias para o funcionamento da aplicação. Abra o arquivo Python em um editor de texto ou IDE e adicione as seguintes linhas no início do código:
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
```
### Passo 2: Função de Pré-Processamento dos Dados
Em seguida, definimos uma função chamada preprocessamento_dados que será responsável por codificar as colunas categóricas do conjunto de dados. A função recebe o DataFrame de dados como entrada e retorna o DataFrame com as colunas categóricas codificadas.

```
def preprocessamento_dados(dados):
    codificador = LabelEncoder()
    dados["profissao"] = codificador.fit_transform(dados["profissao"])
    dados["mix_credito"] = codificador.fit_transform(dados["mix_credito"])
    dados["comportamento_pagamento"] = codificador.fit_transform(dados["comportamento_pagamento"])
    return dados
```    
### Passo 3: Carregando os Dados
Em seguida, carregamos os dados de treinamento e os dados dos novos clientes a serem avaliados a partir de arquivos CSV. Certifique-se de ter os arquivos "clientes.csv" e "novos_clientes.csv" no mesmo diretório que o arquivo Python.

```
tabela = pd.read_csv("clientes.csv")
novos_clientes = pd.read_csv("novos_clientes.csv")
```
### Passo 4: Pré-Processamento dos Dados
Aplicamos a função preprocessamento_dados aos DataFrames tabela e novos_clientes para codificar as colunas categóricas.

```
tabela = preprocessamento_dados(tabela)
novos_clientes = preprocessamento_dados(novos_clientes)
```
### Passo 5: Dividindo os Dados
Agora, dividimos o conjunto de dados de treinamento em recursos (features - x) e o alvo (target - y) para treinamento do modelo.

```
y = tabela["score_credito"]
x = tabela.drop(["id_cliente", "score_credito"], axis=1)
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y)
```
### Passo 6: Inicialização dos Modelos
Inicializamos dois modelos de Machine Learning que usaremos para prever o score de crédito: RandomForestClassifier e KNeighborsClassifier.

```
modelo_arvoredecisao = RandomForestClassifier()
modelo_knn = KNeighborsClassifier()
```
### Passo 7: Treinamento dos Modelos
Agora, treinamos os modelos com os dados de treinamento.


```
modelo_arvoredecisao.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino)
```

### Passo 8: Realizando Previsões e Avaliando os Modelos
Em seguida, fazemos previsões usando os modelos treinados nos dados de teste e avaliamos o desempenho dos modelos usando a métrica de acurácia.

```
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste.to_numpy())

print("Acurácia do RandomForestClassifier:", accuracy_score(y_teste, previsao_arvoredecisao))
print("Acurácia do KNeighborsClassifier:", accuracy_score(y_teste, previsao_knn))
```
### Passo 9: Previsões para Novos Clientes
Finalmente, usamos o modelo RandomForestClassifier para prever o score de crédito dos novos clientes.

```
previsao = modelo_arvoredecisao.predict(novos_clientes)
print(previsao)
```
### Executando a Aplicação
Para executar a aplicação, abra o terminal ou prompt de comando, navegue até o diretório onde o arquivo Python está localizado e execute o seguinte comando:

Copy code
python nomedoarquivo.py
Substitua "nomedoarquivo.py" pelo nome do arquivo Python que contém o código da aplicação.

Com esses passos, você estará apto a executar a aplicação de previsão de score de crédito para novos clientes usando algoritmos de Machine Learning. Caso deseje entender detalhes específicos sobre os algoritmos ou outras funcionalidades utilizadas, você pode consultar a documentação oficial do scikit-learn e pandas para obter informações mais detalhadas.
