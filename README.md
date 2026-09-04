# Predição da Progressão Motora na Doença de Parkinson via Telemonitoramento Vocal

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Visão Geral do Projeto

Este projeto foi desenvolvido como atividade avaliativa da disciplina **PEX0023 - Redes Neurais Artificiais (1106525)**, turma **T01 (2026.2)**, ministrada pela professora **Rosana Cibely Batista Rego**.

O trabalho aborda a aplicação de **Redes Neurais Artificiais**, utilizando uma arquitetura **Perceptron Multicamadas (MLP)** para resolver um problema de regressão relacionado à predição da progressão motora na Doença de Parkinson.

A **Doença de Parkinson (DP)** é uma enfermidade neurodegenerativa progressiva que afeta severamente as funções motoras. A avaliação clínica tradicional depende da escala **UPDRS** (*Unified Parkinson's Disease Rating Scale*), que exige consultas presenciais frequentes.

Neste projeto, uma rede neural MLP é utilizada para predizer o indicador motor da doença (`motor_UPDRS`) a partir de biomarcadores vocais e dados demográficos extraídos do *Parkinson's Telemonitoring Dataset*.

O projeto contempla as etapas de preparação dos dados, padronização das variáveis, construção e treinamento da rede neural, além da avaliação do modelo por meio de métricas de regressão.

---

## Principais Resultados

O modelo supervisionado treinado por meio de uma rede neural densa apresentou bom desempenho de generalização no conjunto de teste.

| Métrica de Avaliação | Valor Obtido | Descrição |
| :--- | :---: | :--- |
| **MAE** (*Mean Absolute Error*) | **2,9924** | Desvio médio absoluto em relação ao score clínico real. |
| **RMSE** (*Root Mean Squared Error*) | **4,2429** | Penaliza erros de maior magnitude. |
| **MSE** (*Mean Squared Error*) | **18,0019** | Erro quadrático médio entre os valores previstos e reais. |
| **R²** (*Coeficiente de Determinação*) | **0,7180** | Aproximadamente **71,80%** da variância do `motor_UPDRS` é explicada pelo modelo. |

---

## Arquitetura do Modelo (MLP)

A rede neural foi desenvolvida utilizando uma arquitetura sequencial composta por três camadas ocultas e uma camada de saída para regressão:

```text
Entrada (19 atributos)
        │
        ▼
Camada Oculta 1 — 64 neurônios — ReLU
        │
        ▼
Camada Oculta 2 — 32 neurônios — ReLU
        │
        ▼
Camada Oculta 3 — 16 neurônios — ReLU
        │
        ▼
Camada de Saída — 1 neurônio — Linear
        │
        ▼
    motor_UPDRS
```

### Detalhamento dos Parâmetros

- **Total de parâmetros treináveis:** 3.905
- **Otimizador:** Adam
- **Taxa de aprendizagem:** 0,001
- **Função de perda:** Erro Quadrático Médio (*MSE*)
- **Número de épocas:** 100
- **Batch size:** 32
- **Pré-processamento:** Padronização dos atributos utilizando Z-score (`StandardScaler`)
- **Divisão dos dados:** 80% para treinamento e 20% para teste

---

## Conjunto de Dados

Foi utilizado o **Parkinson's Telemonitoring Dataset**, disponível por meio do UCI Machine Learning Repository e também distribuído em plataformas como o Kaggle.

O conjunto de dados contém **5.875 registros de gravações biomédicas de voz**, provenientes de **42 pacientes** em estágios iniciais da doença de Parkinson.

O objetivo do modelo é utilizar características vocais e informações demográficas para estimar o valor de `motor_UPDRS`, que representa o indicador motor associado à escala UPDRS.

### Atributos Utilizados

Foram utilizados 19 atributos como entrada do modelo:

- **Demográficos e temporais:** `age`, `sex` e `test_time`
- **Medidas de Jitter (frequência):** `Jitter(%)`, `Jitter(Abs)`, `RAP`, `PPQ5`, `DDP`
- **Medidas de Shimmer (amplitude):** `Shimmer`, `Shimmer(dB)`, `APQ3`, `APQ5`, `APQ11`, `DDA`
- **Razões de ruído vocal:** `HNR` e `NHR`
- **Medidas dinâmicas não lineares:** `RPDE`, `DFA` e `PPE`

---

## Tecnologias Utilizadas

- **Linguagem:** Python 3.x
- **Ambiente de execução:** Google Colab (Jupyter Notebook na nuvem)
- **Processamento de dados:** `pandas`, `numpy`
- **Aprendizado de Máquina e pré-processamento:** `scikit-learn`
- **Deep Learning:** `TensorFlow` / `Keras`
- **Visualização:** `matplotlib`
- **Dataset:** `kagglehub`
- **Editor e documentação:** Overleaf (LaTeX / IEEEtran)

---

## Estrutura do Repositório

```text
├── notebooks/
│   └── parkinson_mlp.ipynb       # Notebook completo executado no Google Colab
├── plots/
│   ├── loss.png                  # Curva de convergência da Loss
│   ├── mae.png                   # Curva de convergência do MAE
│   └── scatter.png               # Gráfico de dispersão Predito vs. Real
├── paper/
│   └── artigo.tex                # Código fonte do artigo em LaTeX (IEEE format)
├── README.md                     # Documentação do projeto
└── requirements.txt              # Dependências do projeto
```

---

## Como Executar no Google Colab

### 1. Abrir o Google Colab

Acesse o [Google Colab](https://colab.research.google.com/).

### 2. Importar o Notebook

Na página inicial do Google Colab:

1. Acesse a opção **Arquivo > Abrir notebook**.
2. Selecione a aba **GitHub**.
3. Insira a URL deste repositório.
4. Selecione o arquivo `notebooks/parkinson_mlp.ipynb`.

Também é possível baixar o arquivo `.ipynb` do repositório e carregá-lo utilizando a opção **Upload de notebook**.

### 3. Instalar as Dependências

Caso necessário, instale as bibliotecas utilizadas no projeto:

```bash
pip install -r requirements.txt
```

### 4. Download Automático do Dataset

O dataset é baixado automaticamente no ambiente da sessão utilizando `kagglehub`:

```python
import kagglehub

# Download da última versão do dataset
path = kagglehub.dataset_download("soroushsaadat81/parkinsons-telemonitoring")

print("Caminho dos arquivos do dataset:", path)
```

### 5. Executar o Notebook

Após o download do dataset, execute as células do notebook em sequência para realizar:

1. Carregamento e exploração dos dados;
2. Seleção das variáveis;
3. Pré-processamento e padronização;
4. Divisão entre treinamento e teste;
5. Construção da rede neural MLP;
6. Treinamento do modelo;
7. Avaliação das métricas;
8. Geração dos gráficos de desempenho.

---

## Avaliação do Modelo

O desempenho foi avaliado utilizando métricas adequadas para problemas de regressão:

- **MAE:** mede o erro absoluto médio entre as previsões e os valores reais.
- **MSE:** calcula o erro quadrático médio, atribuindo maior peso a erros maiores.
- **RMSE:** corresponde à raiz quadrada do MSE e apresenta a mesma unidade da variável de saída.
- **R²:** indica a proporção da variância da variável-alvo explicada pelo modelo.

Os resultados obtidos indicam que a MLP conseguiu aprender uma relação relevante entre os atributos de entrada e o indicador `motor_UPDRS`.

---

## Autora

**Francisca Lorrayne de Lima Santos**

*Graduanda em Engenharia da Computação*

Universidade Federal Rural do Semi-Árido (UFERSA) – Pau dos Ferros/RN

E-mail: santosfranciscalorrayne@gmail.com

---

## Licença

Este projeto está sob a licença MIT. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.
