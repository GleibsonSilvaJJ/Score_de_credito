import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def preprocessamento_dados(dados):
    codificador = LabelEncoder()
    dados["profissao"] = codificador.fit_transform(dados["profissao"])
    dados["mix_credito"] = codificador.fit_transform(dados["mix_credito"])
    dados["comportamento_pagamento"] = codificador.fit_transform(dados["comportamento_pagamento"])
    return dados

# Carregar os Dados
tabela = pd.read_csv("clientes.csv")
novos_clientes = pd.read_csv("novos_clientes.csv")

# Pré-Processamento dos dados
tabela = preprocessamento_dados(tabela)
novos_clientes = preprocessamento_dados(novos_clientes)

# Dividir os dados em features (x) e target (y) para treinamento
y = tabela["score_credito"]
x = tabela.drop(["id_cliente", "score_credito"], axis=1)
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y)

# Inicialização dos modelos
modelo_arvoredecisao = RandomForestClassifier()
modelo_knn = KNeighborsClassifier()

# Treinamento dos modelos
modelo_arvoredecisao.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino)

#Realizar previsões
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste.to_numpy())

# Avaliação dos modelos
print("Acurácia do RandomForestClassifier:",accuracy_score(y_teste, previsao_arvoredecisao))
print("Acurácia do KNeighborsClassifier:", accuracy_score(y_teste, previsao_knn))

#usando o Modelo ForestClassifier para verificar novos clientes
previsao = modelo_arvoredecisao.predict(novos_clientes)
print(previsao)