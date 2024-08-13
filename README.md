
# Algoritmo Preditivo com Keras

Este projeto foi desenvolvido a partir da análise realizada preliminarmente por [vhsmdev](https://github.com/vhsmdev/) no [Projeto de Análise de Dados de Turbinas Eólicas](https://github.com/vhsmdev/analytics-turbina-eolica). Este projeto contudo, tem o objetivo de prever e classificar os limites de potência aceitáveis para as turbinas eólicas, usando um algoritmo de aprendizado classificador. Para o projeto foram usadas as bibliotecas Numpy, Pandas e Tensorflow, para conversão de arrays numéricos, tratamento dos dados e criação do modelo classificador, respectivamente.

### Importando Bibliotecas

Começaremos importando as bibliotecas que usaremos:

```python
import Tensorflow as teras

import pandas as pandinha

from numpy import array as array_numerico
```

A partir do Tensorflow, usaremos a api Keras para criar o nosso modelo de classificação, treinaremos e faremos previsões.

A partir do Pandas, carregaremos os dados, os trataremos e os dividiremos para a etapa de treinamento do modelo.

A partir do Numpy, transformaremos os dados de forma com que o modelo possa interpretá-lo como um array.

### Carregamento e Tratamento dos Dados

O arquivo foi carregado a partir do repositório remoto [raw_turbina_scala.csv](https://raw.githubusercontent.com/vhsmdev/analytics-turbina-eolica/main/data/raw/raw_turbina_scala.csv) do github.

```python
# Carregamento dos Dados
arquivo = pandinha.read_csv(teras.keras.utils.get_file(
    'raw_turbina_scala.csv',
    'https://raw.githubusercontent.com/vhsmdev/analytics-turbina-eolica/main/data/raw/raw_turbina_scala.csv'))
```

e a partir daqui começaremos a tratar os dados,

- A lista a seguir é onde será armazenado o resultado dos limites de potencia aceitáveis

```python
# Definimos a lista "potenciado"
potenciado = []
```

- Trataremos os Dados da Seguinte Forma.
    - Definiremos e Armazenaremos as Potencias Aceitáveis(sendo 1 para Dentro do Limite Aceitável e 0 para Fora do Limite Aceitável).
    - E Adionaremos a Lista "potenciado" como uma nova coluna ao DataSet.

```python
# Armazenamento dos limites aceitáveis
for p, potencia in enumerate(list(arquivo['LV ActivePower (kW)'])):
  if potencia >= (arquivo.loc[p, 'Theoretical_Power_Curve (KWh)'] * 0.95) and potencia <= (arquivo.loc[p, 'Theoretical_Power_Curve (KWh)'] * 1.05):
    potenciado.append(1)

  else:
    potenciado.append(0)

# Criação da coluna "potenciado"
arquivo['potenciado'] = potenciado
```

- Por Fim, Filtraremos os Dados para Que Contenham Números 0 e Os Retiraremos do DataSet, Visto que Eles Podem Criar Vieses e Influenciar Negativamente o Nosso Modelo Durante o Treinamento. Na Falta de Melhor Motivo, Trataremos estes Números 0 como "Momentos de Manutenção".

```python
# Concatenação do DataSet original com o filtro aplicado
arquivo = pandinha.concat([arquivo, arquivo.loc[(arquivo['LV ActivePower (kW)'] == 0.0) |
                                                (arquivo['Wind Speed (m/s)'] == 0.0) |
                                                (arquivo['Theoretical_Power_Curve (KWh)'] == 0.0) |
                                                (arquivo['Wind Direction (°)'] == 0.0)]])

# Exclusão das duplicatas para retirar os números 0
arquivo = arquivo.drop_duplicates(keep = False).drop('Date/Time', axis = 1)
```

agora com os dados tratados, vamos dividi-los entre dados de teste(para o treinamento) e dados de prova(para o teste final).

```python
treino = arquivo[9821:]

prova = arquivo[:9821]

treino_y = treino.pop('potenciado')

prova_y = prova.pop('potenciado')
```

Para a divisão dos dados, foi escolhida uma proporção de 1:3(um para três).

- Antes de Avançarmos, Vamos Criar a Nossa Coluna de Recursos.

```python
colunas_de_recurso = []

for chave in treino.keys():
  colunas_de_recurso.append(teras.feature_column.numeric_column(key = chave))
```

A usaremos mais a frente para a inserção dos dados na primeira camada do modelo.

### Função de Entrada

Para que o modelo reconheça os nossos dados, vamos definir 1(uma) função de entrada, que servirá para a previsão.

```python
# Função de entrada para previsão
def entrada(recursos):
  return teras.data.Dataset.from_tensor_slices(array_numerico(recursos)).batch(256)
```

### Criando o Modelo Classificador

Finalmente chegamos na criação do modelo.

- Criaremos Um Modelo sequencial onde as camadas ficarão empilhadas uma após a outra.

```python
classificador = teras.keras.Sequential([
  teras.keras.layers.Input(shape = (4, )),
```

- A Segunda Camada Estará Totalmente Conectada a Anterior com 24 Neurônios, e Usará a Função de Ativação 'relu' para Introduzir Não-Linearidade em Nosso Modelo.

```python
teras.keras.layers.Dense(24, activation = 'relu'),
```

- A Terceira Camada Será Semelhante a Segunda, Com a Diferença de que Estará Conectada a Anterior com Apenas 12 Neurônios.

```python
teras.keras.layers.Dense(12, activation = 'relu'),
```

- A Última Camada Será a Camada de Saída do Modelo, Contendo Apenas 2 Neurônios, Representando as Saídas que Desejamos(relambrando, as saídas que buscamos são referentes aos valores estarem dentro ou fora da potencia aceitável). Continuando, Vamos Usar a Função de Ativação 'softmax', ao Qual Converterá as Saídas em Probabilidades.

```python
teras.keras.layers.Dense(2, activation = 'softmax')])
```

E assim temos o nosso modelo criado:

```python
classificador = teras.keras.Sequential([
  teras.keras.layers.Input(shape = (4, )),
  teras.keras.layers.Dense(24, activation = 'relu'),
  teras.keras.layers.Dense(12, activation = 'relu'),
  teras.keras.layers.Dense(2, activation = 'softmax')])
```

Com o modelo criado, vamos agora compilá-lo.

```python
classificador.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
```

Usaremos o otimizador 'adam', ele adaptará a taxa de aprendizado durante o processo de aprendizado para cada parâmetro individualmente. A função de perda a ser otimizada durante o treinamento será a 'sparse_categorical_crossentropy', para se calcular a entropia cruzada entre as distribuições de probabilidade das classes verdadeiras e as probabilidades previstas pelo modelo. Por fim, a métrica a ser observada será a precisão do modelo.

### Treinamento e Teste de Precisão

Chegamos na etapa de treinamento e teste do modelo:

```python
classificador.fit(treino, treino_y,
                  epochs = 20,
                  validation_data = (prova, prova_y))
```

Após alguns testes, definimos a quantidade de epochs como 20 para a convergência do modelo, e ao final de cada epoch testaremos o modelo a partir do parâmetro validation_data.

<img src="https://raw.githubusercontent.com/Penkari/recursos/main/Resultado_do_Teste.png">

Ao final a precisão do modelo ficou entre 97% e 95%.

### Previsão e Inserção de Dados

Para a previsão dos dados, precisamos primeiramente definir 2(duas) variáveis, uma para os dados que serão inseridos(variável 'recursado') e a outra para armazenar os valores a serem previstos(variável 'prever').

```python
recursos = ['LV-ActivePower-kW', 'Wind-Speed-m/s', 'Theoretical-Power-Curve-KWh', 'Wind-Direction']
prever = []
```

Agora, para a inserção dos dados, criaremos uma classíca repetição for com a função input:

```python
for recurso in recursado:
  valido = True

  while valido:
    val = input(recurso + ': ')

    if not val.isdigit(): valido = False

  prever.append(float(val))
```

Por fim, usaremos a função de entrada criada anteriormente para inserir os dados da variável 'prever' na função predict e esvaziaremos a lista prever.

```python
previsoes = classificador.predict(entrada([prever]))

prever = []

print(f'Previsto para {funcionamento[list(previsoes[0]).index(max(list(previsoes[0])))]}')
```

Assim a previsão será feita.

## Conclusão

A partir do algoritmo construído, você pode realizar uma simples previsão de dados. É perceptível pelo valor de precisão que a api Keras do Tensorflow é bem poderosa; mesmo com uma grande quantidade de dados a precisão se manteve alta.

A seguir, deixarei alguns exemplos usados para o teste de previsão:

```
380.7 5.4 458.2 11.7 esperado:0 resultado:0

2613.4 11.0 3284.7 72.2 esperado:0 resultado:0

1.0 3.7 84.7 78.5 esperado:0 resultado:0

491.4 5.7 542.1 69.1 esperado:0 resultado:0

2814.8 10.7 3195.4 213.9 esperado:0 resultado:0

3603.0 14.4 3600.0 198.7 esperado:1 resultado:1

835.8 6.8 941.6 190.3 esperado:0 resultado:0

440.8 5.5 471.3 37.9 esperado:0 resultado:0

2374.5 10.1 2884.0 201.7 esperado:0 resultado:0

893.0 9.0 2194.3 26.7 esperado:0 resultado:0
```

- *Eduardo Luiz* ("Dados Visíveis São como Vidros, Faceis de Quebrar e Difíceis de Limpar")