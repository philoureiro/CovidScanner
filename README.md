# Apresentação do Projeto: Classificação de Raios X para COVID-19

## Introdução e Descrição Geral do Projeto
Olá a todos, meu nome é [Seu Nome]. Hoje vou apresentar um projeto que desenvolvemos para processar imagens de raios X de pacientes com e sem COVID-19, utilizando um descritor de histograma de escala de cinza e classificadores de aprendizado de máquina. O objetivo do projeto é identificar automaticamente a presença de COVID-19 em imagens de raios X, analisando as imagens e fazendo previsões.

## Equipe
- Guilheme Antunes Gonçalves dos Santos
- Philipe Borel Loureiro

## Bibliotecas Utilizadas
Utilizamos várias bibliotecas em Python para realizar este projeto:
- `numpy` para manipulação de arrays.
- `scikit-learn` para implementação dos classificadores e pré-processamento.
- `matplotlib` para visualização de dados.
- `tkinter` para interfaces gráficas.
- `time` e `datetime` para medição de tempo e manipulação de datas.

## Estratégia/Metodologia Utilizada para Implementação do Descritor
Implementamos o descritor de histograma de escala de cinza para extrair características das imagens.

### Fluxo de Trabalho:
1. O usuário escolhe o descritor de histograma de cinza no menu.
2. As características das imagens de treino e teste são carregadas de arquivos CSV.
3. Um classificador é treinado com as características de treino.
4. O classificador faz previsões com as características de teste.
5. A precisão do classificador é avaliada e visualizada em uma matriz de confusão.

## Exibição de Partes do Código Fonte com Modificações/Adaptações
### Exemplo Original com MLPClassifier:
```python
def trainMLP(trainData, trainLabels):
    print('[INFO] Training the MLP model...')
    mlp_model = MLPClassifier(random_state=1, hidden_layer_sizes=(5000,), max_iter=1000)
    startTime = time.time()
    mlp_model.fit(trainData, trainLabels)
    elapsedTime = round(time.time() - startTime,2)
    print(f'[INFO] Training done in {elapsedTime}s')
    return mlp_model
```

### Modificação para SGDClassifier:
```python
def trainSGDClassifier(trainData, trainLabels):
    print('[INFO] Training the SGD model...')
    sgd_pipeline = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
    startTime = time.time()
    sgd_pipeline.fit(trainData, trainLabels)
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Training done in {elapsedTime}s')
    return sgd_pipeline
```

## Diferença entre SGD e Ridge Classifiers
### SGD (Stochastic Gradient Descent):
- Método de aprendizado online que atualiza os parâmetros do modelo iterativamente.
- Pode ser usado para grandes conjuntos de dados.
- No nosso projeto, obteve uma precisão de 69% no histograma de cinza das imagens de raios X.

### Ridge Classifier:
- Variante da regressão linear que inclui uma penalização (regularização) para evitar overfitting.
- Geralmente é mais estável e menos suscetível a outliers.
- No nosso projeto, obteve uma precisão de 89% no mesmo conjunto de dados, mostrando uma performance significativamente melhor que o SGD.

## Conclusão
### Resumo:
Utilizamos técnicas avançadas de processamento de imagens e aprendizado de máquina para identificar a presença de COVID-19 em imagens de raios X. Os classificadores diferentes mostraram performances variadas, com o Ridge Classifier apresentando melhores resultados.

### Agradecimento:
Agradeço a todos pela atenção. Estou à disposição para perguntas.
