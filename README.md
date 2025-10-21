# 🧠 MXMH Mental Health Prediction (Análise de Saúde Mental a partir de Hábitos Musicais)

Este projeto utiliza **Machine Learning** para analisar a relação entre hábitos musicais e a **saúde mental**, com base no famoso dataset **"MXMH Survey Results"** do Kaggle.

A aplicação foi desenvolvida em **Python** com **Flask** e **Scikit-Learn**, permitindo prever o risco de condições como **Ansiedade, Depressão, Insônia e TOC** a partir de informações fornecidas pelo usuário (como idade, tempo diário ouvindo música e plataforma de streaming principal).

---

## 🚀 Tecnologias Utilizadas

- **Python 3**
- **Flask** — Framework web leve e rápido
- **Pandas** — Manipulação e análise de dados
- **Scikit-Learn** — Treinamento dos modelos de IA
- **Joblib** — Salvamento e carregamento do modelo treinado
- **HTML5 / CSS3** — Interface web estilizada
- **Bootstrap** — Design moderno e responsivo

---

## 🧩 Modelos de Machine Learning Treinados

Durante o desenvolvimento, foram testados e comparados os seguintes algoritmos:

- 🌲 **Random Forest Classifier**
- 🤖 **Support Vector Machine (SVM)**
- 🧮 **K-Nearest Neighbors (KNN)**
- 📈 **Logistic Regression**

O **Random Forest** apresentou o melhor desempenho médio e foi utilizado na aplicação final em Flask.

---

## 📊 Dataset Utilizado

**Nome:** MXMH Survey Results  
**Fonte:** [Kaggle](https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results)  
**Tamanho:** 10.000+ registros  
**Campos relevantes:**
- Idade (`Age`)
- Plataforma de streaming (`Primary streaming service`)
- Horas ouvindo música (`Hours per day`)
- Gênero favorito (`Fav genre`)
- Níveis de ansiedade, depressão, insônia e TOC

---

## 🧠 Funcionamento da Aplicação

1. O usuário insere suas informações no formulário:
   - Idade  
   - Horas por dia ouvindo música  
   - Plataforma de streaming principal  
   - Gênero musical favorito  

2. O modelo preditivo (Random Forest) processa os dados e estima os **níveis de risco** de cada condição mental.

3. Os resultados são exibidos na tela de forma clara e interativa.

---

## 💡 Objetivo do Projeto

Explorar a relação entre **música e bem-estar mental**, mostrando como dados simples do dia a dia podem ajudar a prever **tendências comportamentais e emocionais**.
