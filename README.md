
# 🦠 Algoritmo de Identificação de Paciente com COVID-19 Utilizando Inteligência Artificial

## 📚 Disciplina: Processamento de Imagens
## 💻 Curso: Engenharia de Software
## 🎓 Universidade Tecnológica Federal do Paraná - Cornélio Procópio

## 👥 Equipe
- Guilherme Antunes Gonçalves dos Santos
- Philipe Borel Loureiro

## 🛠️ Bibliotecas Utilizadas
Utilizamos várias bibliotecas em Python para realizar este projeto:
- `numpy` para manipulação de arrays.
- `scikit-learn` para implementação dos classificadores e pré-processamento.
- `matplotlib` para visualização de dados.
- `tkinter` para interfaces gráficas.
- `time` e `datetime` para medição de tempo e manipulação de datas.

## 📝 Estratégia/Metodologia Utilizada para Implementação do Descritor
Implementamos o descritor de histograma de escala de cinza para extrair características das imagens.

### 🔄 Fluxo de Trabalho:
1. O usuário escolhe o descritor de histograma de cinza no menu.
2. As características das imagens de treino e teste são carregadas de arquivos CSV.
3. Um classificador é treinado com as características de treino.
4. O classificador faz previsões com as características de teste.
5. A precisão do classificador é avaliada e visualizada em uma matriz de confusão.

## 💻 Exibição de Partes do Código Fonte com Modificações/Adaptações
### Exemplo Original com MLPClassifier:
```python
def trainMLP(trainData, trainLabels):
    print('[INFO] Training the MLP model...')
    mlp_model = MLPClassifier(random_state=1, hidden_layer_sizes=(5000,), max_iter=1000)
    startTime = time.time()
    mlp_model.fit(trainData, trainLabels)
    elapsedTime = round(time.time() - startTime, 2)
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

### Modificação para RidgeClassifier:
```python
def trainRidgeClassifier(trainData, trainLabels):
    print('[INFO] Training the Ridge Classifier model...')
    ridge_classifier = RidgeClassifier()
    startTime = time.time()
    ridge_classifier.fit(trainData, trainLabels)
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Training done in {elapsedTime}s')
    return ridge_classifier
```

## 🔍 Diferença entre SGD e Ridge Classifiers
### 📉 SGD (Stochastic Gradient Descent):
- Método de aprendizado online que atualiza os parâmetros do modelo iterativamente.
- Pode ser usado para grandes conjuntos de dados.
- No nosso projeto, obteve uma precisão de 69% no histograma de cinza das imagens de raios X.

### 📈 Ridge Classifier:
- Variante da regressão linear que inclui uma penalização (regularização) para evitar overfitting.
- Geralmente é mais estável e menos suscetível a outliers.
- No nosso projeto, obteve uma precisão de 89% no mesmo conjunto de dados, mostrando uma performance significativamente melhor que o SGD.

## 📊 Conclusão
### 📌 Resumo:
Utilizamos técnicas avançadas de processamento de imagens e aprendizado de máquina para identificar a presença de COVID-19 em imagens de raios X. Os classificadores diferentes mostraram performances variadas, com o Ridge Classifier apresentando melhores resultados.

🔗 Para mais detalhes, acesse o repositório do projeto: [CovidScanner](https://github.com/philoureiro/CovidScanner)

## 🏁 Instruções para Rodar o Projeto

### 📥 Pré-requisitos:
1. **Python 3.x** instalado.
2. **Miniconda** instalado. Baixe e instale o Miniconda seguindo as instruções neste link: [Miniconda Installation](https://docs.anaconda.com/free/miniconda/).

### 📦 Criar e Configurar o Ambiente Conda:
1. Abra o terminal ou prompt de comando.
2. Crie um novo ambiente chamado `CovidScanner` e instale as bibliotecas necessárias usando o seguinte comando:
    ```sh
    conda create -n CovidScanner -c defaults -c conda-forge python=3.8 pip numpy matplotlib progress scikit-learn pillow
    ```
3. Ative o ambiente:
    ```sh
    conda activate CovidScanner
    ```

### 📂 Clonar o Repositório:
```sh
git clone https://github.com/philoureiro/CovidScanner.git
cd CovidScanner
```

### 🚀 Executar o Projeto:
1. Navegue até o diretório `classifiers`:
    ```sh
    cd classifiers
    ```
2. Para rodar todos os classificadores, execute o seguinte comando:
    ```sh
    python run_all_classifier.py
    ```
3. Caso queira experimentar apenas um dos modelos de classificação, você pode executar individualmente qualquer um dos arquivos dentro da pasta `classifiers`:
    ```sh
    python ridge_classifier.py
    ```
    ou
    ```sh
    python sgd_classifier.py
    ```

🎉 Agora você está pronto para rodar o projeto e explorar as capacidades do algoritmo de identificação de paciente com COVID-19!

