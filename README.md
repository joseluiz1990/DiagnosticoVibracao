# ⚙️ Classificação de Falhas em Máquinas Rotativas com ML Clássico e Redes Neurais (MLP)

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Este repositório contém a implementação metodológica para classificação automática de falhas operacionais em motores elétricos de indução de $45\text{ kW}$ a partir da análise de sinais vibracionais triaxiais.

O projeto combina **Processamento Digital de Sinais (DSP)**, **Engenharia de Atributos Híbrida** (tempo e frequência) e uma estratégia rigorosa de **pré-processamento blindado contra vazamento de dados (*Data Leakage*)**, permitindo comparar classificadores clássicos (**SVM**, **KNN**) com uma **Rede Neural Perceptron Multicamadas (MLP)**.

---

## 📌 Principais Destaques & Resultados

* **Fusão Vetorial:** Redução da dimensionalidade espacial dos eixos $X, Y, Z$ em um vetor único de magnitude via **Resultante Euclidiana**.
* **Prevenção de Data Leakage:** Trata outliers e aplica escalonamento Z-Score ajustados *exclusivamente* no conjunto de treino.
* **Modelo MLP (Rede Neural):** Atingiu **100% de acurácia global**, especificidade de $100\%$ e sensibilidade de $100\%$ no conjunto de teste independente (66 amostras inéditas), com $\text{AUC-ROC} = 1,0000$.
* **Modelo SVM (Kernel RBF):** Alcançou **96,97% de acurácia** no teste final ($\text{AUC-ROC} = 0,9982$), demonstrando alta viabilidade para implementação em **Edge Computing** (hardware embarcado de baixo custo).

---

## 📐 Metodologia & Pipeline

O fluxo de processamento de dados é dividido em 7 etapas principais:

```
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
│ 1. Sinal Bruto  │ --> │ 2. Resultante        │ --> │ 3. Extração de         │
│    (3 Eixos)    │     │    Euclidiana        │     │    Atributos           │
└─────────────────┘     └──────────────────────┘     └────────────────────────┘
                                                                 │
┌─────────────────┐     ┌──────────────────────┐                 ▼
│ 6. Modelagem &  │ <-- │ 5. Divisão & Blindagem│ <-- ┌────────────────────────┐
│    Otimização   │     │    (Treino 70%/30%)  │     │ 4. Análise Exploratória│
└─────────────────┘     └──────────────────────┘     └────────────────────────┘
         │
         ▼
┌─────────────────┐
│ 7. Validação    │
│    no Teste     │
└─────────────────┘
```

### 1. Processamento Digital de Sinais (DSP) & Atributos
* **Fusão Triaxial:**
  $$R(t) = \sqrt{X(t)^2 + Y(t)^2 + Z(t)^2}$$
* **Domínio do Tempo:**
  * **RMS:** Severidade global de energia vibratória.
  * **Curtose:** Medida de impulsividade e choques cíclicos.
  * **Fator de Crista:** Razão entre valor de pico e RMS.
* **Domínio da Frequência (FFT + Janela de Hanning):**
  * Amplitude da frequência fundamental **1X** ($\sim 49,8\text{ Hz}$).
  * Amplitude da 2ª harmônica **2X** ($\sim 99,4\text{ Hz}$).
  * Amplitude da 3ª harmônica **3X** ($\sim 149,2\text{ Hz}$).

### 2. Tratamento Estatístico
* **Detecção de Outliers:** Método da Amplitude Interquartílica (IQR) aplicado unicamente sobre a classe saudável do treino ($1,5 \times \text{IQR}$).
* **Padronização:** Z-Score ($Mean = 0, Std = 1$).

---

## 📊 Desempenho dos Modelos

### Validação Cruzada (K-Fold = 5) - Conjunto de Treino
| Modelo | Acurácia Média (%) | Desvio Padrão (%) |
| :--- | :---: | :---: |
| **SVM (Kernel RBF)** | **98,67%** | $\pm 1,63\%$ |
| **MLP (2 Camadas: 64x32)** | **97,98%** | $\pm 1,65\%$ |
| **KNN (k=5, Euclidean)** | **97,33%** | $\pm 2,49\%$ |
| Árvore de Decisão | 93,29% | $\pm 3,65\%$ |
| Random Forest | 93,29% | $\pm 4,71\%$ |
| Regressão Logística | 89,29% | $\pm 3,84\%$ |

### Avaliação Final no Conjunto de Teste Independente (30% - 66 Amostras)
| Métrica | KNN | SVM | MLP (RNA) |
| :--- | :---: | :---: | :---: |
| **Acurácia Global** | 95,45% | 96,97% | **100,00%** |
| **Precisão (Falha)** | 97,06% | 97,14% | **100,00%** |
| **Recall / Sensibilidade (Falha)** | 94,29% | 97,14% | **100,00%** |
| **F1-Score (Falha)** | 0,9565 | 0,9714 | **1,0000** |
| **AUC-ROC** | 0,9982 | 0,9982 | **1,0000** |
| **IC 95% AUC-ROC (Bootstrap)** | [0.9916; 1.0000] | [0.9916; 1.0000] | **[1.0000; 1.0000]** |

---

## 📁 Estrutura do Repositório

```text
├── data/
│   ├── raw/                 # Dados brutos (.mat - Kaggle Vibration Faults Dataset)
│   └── processed/           # Dataset tabular com atributos extraídos (220 amostras)
├── notebooks/
│   └── pipeline_falhas.ipynb # Notebook Colab com pré-processamento, EDA e modelos
├── src/
│   ├── dsp.py               # Funções de Resultante Euclidiana e FFT
│   ├── feature_extraction.py# Extração de RMS, Curtose, Crista e Harmônicas
│   └── models.py            # Treinamento e avaliação de SVM, KNN e PyTorch MLP
├── README.md
└── requirements.txt
```

---

## 🛠️ Instalação e Execução

### Pré-requisitos
* Python 3.10+
* Google Colab ou ambiente Python local

### 1. Clonar o Repositório
```bash
git clone https://github.com/usuario/classificacao-falhas-maquinas-rotativas.git
cd classificacao-falhas-maquinas-rotativas
```

### 2. Instalar Dependências
```bash
pip install -r requirements.txt
```

*Principais bibliotecas requeridas:*
`numpy`, `pandas`, `scipy`, `scikit-learn`, `torch`, `matplotlib`, `seaborn`

---

## 👥 Autores & Informações Acadêmicas

* **José Luiz Lopes** — *jose.lopes5@aluno.cps.gov.sp.br*

**Instituição:** FATEC Dep. Ari Fossen / Centro Paula Souza — Jundiaí–SP

---

## 📜 Licença

Este projeto está licenciado sob a Licença MIT - consulte o arquivo [LICENSE](LICENSE) para obter mais detalhes.
