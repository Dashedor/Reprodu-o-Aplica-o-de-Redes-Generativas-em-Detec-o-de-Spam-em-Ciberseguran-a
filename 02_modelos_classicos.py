"""Modelos clássicos (SVM, RF, NB, RL, KNN) x vetorização (BoW, TF-IDF).

Você informa o(s) CSV(s) de treino e o(s) CSV(s) de teste (benignos ou envenenados). Cada treino é
combinado com cada teste.

Padrão (como no artigo): hiperparâmetros default do scikit-learn.
  python 02_modelos_classicos.py --treino saida/treino.csv saida/treino_gpt4o.csv --teste saida/teste.csv

Com otimização (--otimizar): GridSearchCV (validação cruzada) SÓ no treino benigno (--base; padrão = 1º
--treino), nunca no teste. Os melhores parâmetros são salvos em CSV e reutilizados em TODOS os treinos,
para que a diferença entre cenários venha do envenenamento e não de ajustes diferentes.
  python 02_modelos_classicos.py --otimizar --treino saida/treino.csv saida/treino_gpt4o.csv --teste saida/teste.csv

Para reaproveitar parâmetros já encontrados, sem refazer a busca:
  python 02_modelos_classicos.py --params saida/melhores_parametros.csv --treino ... --teste ...
"""
import argparse
import json
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from common import SAIDA, carregar_csv, nome_do_arquivo, salvar_resultados

warnings.filterwarnings("ignore")

VETORIZADORES = {"BoW": CountVectorizer, "TF-IDF": TfidfVectorizer}

# Grades de busca (prefixo "clf__" = modelo; "vec__" = vetorizador)
GRADE_MODELOS = {
    "Regressão Logística": {"clf__C": [0.1, 1, 10, 100], "clf__class_weight": [None, "balanced"]},
    "Naive Bayes": {"clf__alpha": [0.1, 0.5, 1.0]},
    "Random Forest": {"clf__n_estimators": [100, 300], "clf__min_samples_leaf": [1, 2]},
    "KNN": {"clf__n_neighbors": [3, 5, 11], "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "cosine"]},
    "SVM": {"clf__kernel": ["linear", "rbf"], "clf__C": [0.1, 1, 10]},
}
GRADE_VEC = {"vec__ngram_range": [(1, 1), (1, 2)]}
GRADE_VEC_TFIDF_EXTRA = {"vec__sublinear_tf": [False, True]}


def criar_modelos(seed, knn_cosseno=False, rapido=False):
    """rapido=True: SVM sem probability (só para a busca, que usa F1)."""
    return {
        "Regressão Logística": LogisticRegression(max_iter=1000, random_state=seed),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1 if rapido else -1),
        "KNN": (KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance") if knn_cosseno
                else KNeighborsClassifier(n_neighbors=5)),
        "SVM": SVC(probability=not rapido, random_state=seed),
    }


def buscar_parametros(base, args):
    """GridSearchCV por (modelo, vetorização) no treino benigno. Retorna {(modelo, vec): {'vec': {...}, 'clf': {...}}}."""
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    melhores, linhas = {}, []
    for vec_nome, Vec in VETORIZADORES.items():
        for nome, modelo in criar_modelos(args.seed, rapido=True).items():
            grade = {**GRADE_MODELOS[nome], **GRADE_VEC, **(GRADE_VEC_TFIDF_EXTRA if vec_nome == "TF-IDF" else {})}
            pipe = Pipeline([("vec", Vec()), ("clf", modelo)])
            busca = GridSearchCV(pipe, grade, cv=cv, scoring=args.scoring, n_jobs=-1)
            busca.fit(base["CONTENT"], base["CLASS"])
            bp = busca.best_params_
            melhores[(nome, vec_nome)] = {
                "vec": {k[5:]: v for k, v in bp.items() if k.startswith("vec__")},
                "clf": {k[5:]: v for k, v in bp.items() if k.startswith("clf__")}}
            linhas.append({"Modelo": nome, "Vetorizacao": vec_nome, f"CV_{args.scoring}": round(busca.best_score_, 4),
                           "Parametros": json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in bp.items()})})
            print(f"  [{vec_nome:6s}] {nome:20s} CV-{args.scoring}={busca.best_score_:.3f}  {bp}")
    pd.DataFrame(linhas).to_csv(args.saida_params, index=False)
    print(f"Melhores parâmetros salvos em {args.saida_params}")
    return melhores


def carregar_parametros(caminho):
    melhores = {}
    for _, r in pd.read_csv(caminho).iterrows():
        bp = json.loads(r["Parametros"])
        if "vec__ngram_range" in bp:
            bp["vec__ngram_range"] = tuple(bp["vec__ngram_range"])
        melhores[(r["Modelo"], r["Vetorizacao"])] = {
            "vec": {k[5:]: v for k, v in bp.items() if k.startswith("vec__")},
            "clf": {k[5:]: v for k, v in bp.items() if k.startswith("clf__")}}
    return melhores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treino", nargs="+", default=[str(SAIDA / "treino.csv")], help="um ou mais CSVs de treino benigno")
    ap.add_argument("--teste", nargs="+", default=[str(SAIDA / "teste.csv"), str(SAIDA / "teste_GPT-6.csv"), str(SAIDA / "teste_gemini-3.1pro.csv")], help="um ou mais CSVs de teste benignos ou envenenados")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--knn-cosseno", action="store_true", help="KNN com distância cosseno e pesos por distância (sem --otimizar)")
    ap.add_argument("--otimizar", action="store_true", help="busca de hiperparâmetros com validação cruzada no treino benigno")
    ap.add_argument("--base", help="CSV de treino benigno usado na busca (padrão: 1º --treino)")
    ap.add_argument("--cv", type=int, default=5, help="nº de folds da validação cruzada")
    ap.add_argument("--scoring", default="f1", help="métrica da busca: f1, roc_auc, accuracy, recall...")
    ap.add_argument("--params", help="CSV de melhores parâmetros já encontrados (pula a busca)")
    ap.add_argument("--saida-params", default=str(SAIDA / "melhores_parametros.csv"))
    ap.add_argument("--saida", default=None)
    args = ap.parse_args()

    melhores = None
    if args.params:
        melhores = carregar_parametros(args.params)
        print(f"Usando parâmetros de {args.params}")
    elif args.otimizar:
        base = carregar_csv(args.base or args.treino[0])
        print(f"Buscando hiperparâmetros em {args.base or args.treino[0]} ({len(base)} linhas, cv={args.cv}, scoring={args.scoring})")
        melhores = buscar_parametros(base, args)
    config = "otimizado" if melhores else "padrao"
    saida = args.saida or str(SAIDA / ("resultados_classicos_otimizado.csv" if melhores else "resultados_classicos.csv"))

    testes = {nome_do_arquivo(p): carregar_csv(p) for p in args.teste}
    linhas = []
    for p_tr in args.treino:
        tr, nome_tr = carregar_csv(p_tr), nome_do_arquivo(p_tr)
        for vec_nome, Vec in VETORIZADORES.items():
            for nome, modelo in criar_modelos(args.seed, args.knn_cosseno).items():
                vec = Vec(**(melhores[(nome, vec_nome)]["vec"] if melhores else {}))
                if melhores:
                    modelo.set_params(**melhores[(nome, vec_nome)]["clf"])
                Xtr = vec.fit_transform(tr["CONTENT"])
                modelo.fit(Xtr, tr["CLASS"])
                for nome_te, te in testes.items():
                    Xte = vec.transform(te["CONTENT"])
                    pred, prob = modelo.predict(Xte), modelo.predict_proba(Xte)[:, 1]
                    linhas.append({
                        "Modelo": nome, "Config": config, "Treino": nome_tr, "Teste": nome_te, "Vetorizacao": vec_nome,
                        "Acc": accuracy_score(te["CLASS"], pred),
                        "Precision": precision_score(te["CLASS"], pred, zero_division=0),
                        "Recall": recall_score(te["CLASS"], pred, zero_division=0),
                        "F1": f1_score(te["CLASS"], pred, zero_division=0),
                        "ROC_AUC": roc_auc_score(te["CLASS"], prob)})
    df = pd.DataFrame(linhas)
    salvar_resultados(df, saida, index=["Modelo", "Treino", "Teste"], columns="Vetorizacao")


if __name__ == "__main__":
    main()
