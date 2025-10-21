from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import traceback

app = Flask(__name__)

# ---------- FUNÇÃO: treinar modelos (executa na inicialização) ----------
def train_models(csv_path="mxmh_survey_results.csv"):
    dados = pd.read_csv(csv_path)
    # Escolhe colunas conforme seu script original
    colunas = ['Age', 'Primary streaming service', 'Hours per day',
               'Fav genre', 'Anxiety', 'Depression', 'Insomnia', 'OCD']
    dados = dados[colunas].dropna()
    dados['Age'] = pd.to_numeric(dados['Age'], errors='coerce')
    dados['Hours per day'] = pd.to_numeric(dados['Hours per day'], errors='coerce')
    dados = dados.dropna()

    encoders = {}
    # Criamos encoders para colunas categóricas (inclui doenças para inversão)
    for col in ['Primary streaming service', 'Fav genre', 'Anxiety', 'Depression', 'Insomnia', 'OCD']:
        enc = LabelEncoder()
        dados[col] = enc.fit_transform(dados[col])
        encoders[col] = enc

    # Features e doenças
    X = dados[['Age', 'Primary streaming service', 'Hours per day']]
    doencas = ['Anxiety', 'Depression', 'Insomnia', 'OCD']

    modelos_rf = {}
    modelos_svm = {}
    modelos_knn = {}
    modelos_lr = {}
    acuracias = {}

    # Treina um classificador por doença e guarda acurácias
    for doenca in doencas:
        y = dados[doenca]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

        modelo_rf = RandomForestClassifier(random_state=50)
        modelo_rf.fit(X_train, y_train)
        modelos_rf[doenca] = modelo_rf
        y_pred_rf = modelo_rf.predict(X_test)
        acc_rf = accuracy_score(y_test, y_pred_rf)

        modelo_svm = SVC(random_state=42)
        modelo_svm.fit(X_train, y_train)
        modelos_svm[doenca] = modelo_svm
        y_pred_svm = modelo_svm.predict(X_test)
        acc_svm = accuracy_score(y_test, y_pred_svm)

        modelo_knn = KNeighborsClassifier(n_neighbors=5)
        modelo_knn.fit(X_train, y_train)
        modelos_knn[doenca] = modelo_knn
        y_pred_knn = modelo_knn.predict(X_test)
        acc_knn = accuracy_score(y_test, y_pred_knn)

        modelo_lr = LogisticRegression(max_iter=1000, random_state=42)
        modelo_lr.fit(X_train, y_train)
        modelos_lr[doenca] = modelo_lr
        y_pred_lr = modelo_lr.predict(X_test)
        acc_LR = accuracy_score(y_test, y_pred_lr)

        acuracias[doenca] = {
            'RandomForest': acc_rf,
            'SVM': acc_svm,
            'KNN': acc_knn,
            'LR': acc_LR
        }

    # Determina o melhor modelo por doença (por acurácia)
    melhor_modelo_nomes = {}
    for doenca, metricas in acuracias.items():
        melhor_nome = max(metricas, key=metricas.get)
        melhor_modelo_nomes[doenca] = melhor_nome

    return {
        "encoders": encoders,
        "modelos_rf": modelos_rf,
        "modelos_svm": modelos_svm,
        "modelos_knn": modelos_knn,
        "modelos_lr": modelos_lr,
        "acuracias": acuracias,
        "melhor_modelo_nomes": melhor_modelo_nomes
    }

# Treina tudo ao iniciar (pode demorar alguns segundos dependendo do dataset)
try:
    store = train_models("mxmh_survey_results.csv")
    print("✅ Modelos treinados com sucesso.")
except Exception as e:
    print("❌ Erro ao treinar modelos:", e)
    traceback.print_exc()
    store = None

# ---------- ROTAS ----------
@app.route('/', methods=['GET', 'POST'])
def index():
    if store is None:
        return "Erro: modelos não estão disponíveis. Veja logs no terminal."

    encoders = store['encoders']
    modelos_rf = store['modelos_rf']
    modelos_svm = store['modelos_svm']
    modelos_knn = store['modelos_knn']
    modelos_lr = store['modelos_lr']
    acuracias = store['acuracias']
    melhor_modelo_nomes = store['melhor_modelo_nomes']

    # Opções para selects (originais antes do label-encode)
    platform_options = list(encoders['Primary streaming service'].classes_)
    genre_options = list(encoders['Fav genre'].classes_)

    resultado_final = None
    resultados_por_modelo = {}
    erros = None

    if request.method == 'POST':
        try:
            # Ler inputs do formulário
            age = float(request.form.get('age', 25))
            hours = float(request.form.get('hours', 2.0))
            # platform: campo contém o texto original (ex: "Spotify") — precisamos transformar no código usado pelo encoder
            platform_raw = request.form.get('platform')
            if platform_raw is None:
                platform_raw = platform_options[0]
            platform_code = int(encoders['Primary streaming service'].transform([platform_raw])[0])

            # Monta entrada no formato (Age, Primary streaming service, Hours per day)
            entrada = np.array([[age, platform_code, hours]])

            doencas = ['Anxiety', 'Depression', 'Insomnia', 'OCD']
            resultados = {}
            # Melhor modelo por doença
            for doenca in doencas:
                melhor_nome = melhor_modelo_nomes[doenca]
                if melhor_nome == 'RandomForest':
                    modelo = modelos_rf[doenca]
                elif melhor_nome == 'SVM':
                    modelo = modelos_svm[doenca]
                elif melhor_nome == 'KNN':
                    modelo = modelos_knn[doenca]
                else:
                    modelo = modelos_lr[doenca]

                pred = modelo.predict(entrada)[0]
                classe = encoders[doenca].inverse_transform([pred])[0]
                resultados[doenca] = {
                    "classe": classe,
                    "modelo_usado": melhor_nome
                }

            # Previsões separadas por modelo (para mostrar ao usuário)
            resultados_por_modelo = {
                "RandomForest": {},
                "SVM": {},
                "KNN": {},
                "LR": {}
            }
            for doenca in doencas:
                # RF
                pred = modelos_rf[doenca].predict(entrada)[0]
                resultados_por_modelo["RandomForest"][doenca] = encoders[doenca].inverse_transform([pred])[0]
                # SVM
                pred = modelos_svm[doenca].predict(entrada)[0]
                resultados_por_modelo["SVM"][doenca] = encoders[doenca].inverse_transform([pred])[0]
                # KNN
                pred = modelos_knn[doenca].predict(entrada)[0]
                resultados_por_modelo["KNN"][doenca] = encoders[doenca].inverse_transform([pred])[0]
                # LR
                pred = modelos_lr[doenca].predict(entrada)[0]
                resultados_por_modelo["LR"][doenca] = encoders[doenca].inverse_transform([pred])[0]

            # Gravidade (transforma classes em índices para ordenar)
            gravidade = {}
            for doenca, info in resultados.items():
                indice = int(encoders[doenca].transform([info['classe']])[0])
                gravidade[doenca] = indice
            gravidade_ordenada = sorted(gravidade.items(), key=lambda x: x[1], reverse=True)
            maior_risco_doenca, maior_risco_nivel = gravidade_ordenada[0]
            resultado_final = {
                "resultados": resultados,
                "maior_risco_doenca": maior_risco_doenca,
                "maior_risco_nivel": resultados[maior_risco_doenca]['classe']
            }

        except Exception as e:
            erros = str(e)
            traceback.print_exc()

    return render_template(
        'index.html',
        platform_options=platform_options,
        genre_options=genre_options,
        resultado_final=resultado_final,
        resultados_por_modelo=resultados_por_modelo,
        acuracias=acuracias,
        erros=erros
    )

if __name__ == '__main__':
    app.run(debug=True)
