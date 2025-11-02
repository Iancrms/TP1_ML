"""
run_optimization_report.py (versão completa final - atualizada)

Executa otimização de hiperparâmetros com RandomizedSearchCV para RF e SVM,
gera métricas funcionais (Recall, F1) e não-funcionais (tempo, tamanho),
gráficos comparativos e relatório TXT.
Inclui gráfico de trade-off F1 x Tempo de treino.
"""

import os
import json
import time
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

from scipy.stats import randint, uniform

# Definir diretório de saída
OUT_DIR = "./resultados"
os.makedirs(OUT_DIR, exist_ok=True)

# Carregar datasets
def load_or_create(path, n_samples, n_features, seed, dataset_name_label):
    if os.path.exists(path):
        df = pd.read_csv(path)
        X = df.drop('Result', axis=1)
        y = df['Result']
        return X, y, dataset_name_label
    else:
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features,
            n_informative=max(2, n_features//3), n_redundant=1,
            n_classes=2, random_state=seed
        )
        X = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])
        y = pd.Series(y)
        return X, y, f"SYNTHETIC_{dataset_name_label}"

path_d1 = "./PhishingWebsites11k.csv"
path_d2 = "./WebsitePhishing1k.csv"

X1, y1, name1 = load_or_create(path_d1, 4000, 25, 42, "PhishingWebsites11k")
X2, y2, name2 = load_or_create(path_d2, 1200, 20, 1, "WebsitePhishing1k")

# Distribuições de hiperparâmetros (grid)
rf_param_dist = {
    'n_estimators': randint(50, 600),
    'max_depth': [None, 5, 10, 20, 30],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': randint(2, 10)
}

svm_param_dist = {
    'C': uniform(0.01, 100),
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# Métricas
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='macro', zero_division=0),
    'recall': make_scorer(recall_score, average='macro', zero_division=0),
    'f1': make_scorer(f1_score, average='macro', zero_division=0)
}

# Validação cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Otimização pra coleta de métricas
def run_randomized(model, param_dist, X, y, model_name, dataset_name, n_iter=20):
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        refit=False,
        cv=cv,
        n_jobs=-1,
        random_state=42,
        return_train_score=True
    )
    t0 = time.time()
    search.fit(X, y)
    total_time = time.time() - t0

    results = search.cv_results_
    df = pd.DataFrame(results)

    metric_map = {
        'mean_test_accuracy': 'accuracy',
        'mean_test_precision': 'precision',
        'mean_test_recall': 'recall',
        'mean_test_f1': 'f1'
    }
    for old, new in metric_map.items():
        if old in df.columns:
            df[new] = df[old]

    # Medir tempo e tamanho
    fit_times = []
    pred_times = []
    model_sizes = []

    print(f"[{dataset_name}] {model_name}: Medindo tempo e tamanho para {n_iter} configurações...")

    for params in df['params']:
        clf = model.set_params(**params)
        # Tempo de treino
        t0 = time.time()
        clf.fit(X, y)
        fit_times.append(time.time() - t0)

        # Tempo de predição (média de 10 execuções)
        t0 = time.time()
        for _ in range(10):
            clf.predict(X)
        pred_times.append((time.time() - t0) / 10)

        # Tamanho do modelo
        model_sizes.append(sys.getsizeof(pickle.dumps(clf)))

    df['mean_fit_time'] = fit_times
    df['mean_predict_time'] = pred_times
    df['model_size_bytes'] = model_sizes

    # Melhor configuração por F1
    best_idx = df['f1'].idxmax()
    best_row = df.iloc[best_idx]
    best_model = model.set_params(**best_row['params'])
    best_model.fit(X, y)

    info = {
        "best_params": best_row['params'],
        "best_f1": best_row['f1'],
        "fit_time": best_row['mean_fit_time'],
        "predict_time": best_row['mean_predict_time'],
        "size_bytes": best_row['model_size_bytes'],
        "total_search_time": total_time
    }

    print(f"[{dataset_name}] {model_name}: Melhor F1 = {info['best_f1']:.4f}")
    return df, info

# Executar otimizações
rf_d1, info_rf_d1 = run_randomized(RandomForestClassifier(random_state=42), rf_param_dist, X1, y1, "RandomForest", name1)
svm_d1, info_svm_d1 = run_randomized(SVC(), svm_param_dist, X1, y1, "SVM", name1)
rf_d2, info_rf_d2 = run_randomized(RandomForestClassifier(random_state=42), rf_param_dist, X2, y2, "RandomForest", name2)
svm_d2, info_svm_d2 = run_randomized(SVC(), svm_param_dist, X2, y2, "SVM", name2)

# Salvar em csv
rf_d1.to_csv(os.path.join(OUT_DIR, f"rf_results_{name1}.csv"), index=False)
svm_d1.to_csv(os.path.join(OUT_DIR, f"svm_results_{name1}.csv"), index=False)
rf_d2.to_csv(os.path.join(OUT_DIR, f"rf_results_{name2}.csv"), index=False)
svm_d2.to_csv(os.path.join(OUT_DIR, f"svm_results_{name2}.csv"), index=False)

# Funções para os plots
def plot_metric(df, metric, model_name, dataset_name, ylabel=None):
    if metric not in df.columns:
        return
    df_sorted = df.sort_values('f1', ascending=False).reset_index(drop=True)
    plt.figure(figsize=(9, 5))
    plt.plot(range(len(df_sorted)), df_sorted[metric], marker='o', markersize=4, alpha=0.7, color='tab:blue')
    plt.title(f"{model_name} - {ylabel or metric} ({dataset_name})")
    plt.xlabel("Configuração (ordenada por F1 decrescente)")
    plt.ylabel(ylabel or metric)
    plt.grid(True, alpha=0.3)
    path = os.path.join(OUT_DIR, f"{model_name}_{dataset_name}_{metric}.png")
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()

def plot_comparison(df_rf, df_svm, metric, dataset_name, ylabel=None):
    n = min(len(df_rf), len(df_svm), 20)
    df_rf = df_rf.sort_values('f1', ascending=False).head(n)
    df_svm = df_svm.sort_values('f1', ascending=False).head(n)

    plt.figure(figsize=(9, 5))
    plt.plot(range(n), df_rf[metric], label='Random Forest', marker='o', color='orange')
    plt.plot(range(n), df_svm[metric], label='SVM', marker='s', color='blue')
    plt.legend()
    plt.title(f"RF vs SVM - {ylabel or metric} ({dataset_name})")
    plt.xlabel("Top Configurações")
    plt.ylabel(ylabel or metric)
    plt.grid(True, alpha=0.3)
    path = os.path.join(OUT_DIR, f"Compare_RF_SVM_{dataset_name}_{metric}.png")
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()

def plot_f1_vs_time(df, model_name, dataset_name):
    df_sorted = df.sort_values('f1', ascending=False)
    plt.figure(figsize=(8, 6))
    plt.scatter(df_sorted['mean_fit_time'], df_sorted['f1'], alpha=0.7, s=60, color='tab:purple')
    plt.title(f"{model_name} ({dataset_name})\nTrade-off: F1 × Tempo de Treino")
    plt.xlabel("Tempo médio de treino (s)")
    plt.ylabel("F1-score médio (validação)")
    plt.grid(True, alpha=0.3)
    path = os.path.join(OUT_DIR, f"{model_name}_{dataset_name}_F1_vs_Time.png")
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()

# Métricas para os gráficos
metrics_to_plot = [
    ('recall', 'Recall Médio'),
    ('f1', 'F1-Score Médio'),
    ('mean_fit_time', 'Tempo Médio de Treino (s)'),
    ('model_size_bytes', 'Tamanho do Modelo (bytes)')
]

# Gráficos individuais
for metric, ylabel in metrics_to_plot:
    plot_metric(rf_d1, metric, "RF", name1, ylabel)
    plot_metric(svm_d1, metric, "SVM", name1, ylabel)
    plot_metric(rf_d2, metric, "RF", name2, ylabel)
    plot_metric(svm_d2, metric, "SVM", name2, ylabel)

# Gráficos RF vs SVM (top 20)
for metric, ylabel in metrics_to_plot:
    plot_comparison(rf_d1, svm_d1, metric, name1, ylabel)
    plot_comparison(rf_d2, svm_d2, metric, name2, ylabel)

# Gráfico F1 x Tempo
plot_f1_vs_time(rf_d1, "Random Forest", name1)
plot_f1_vs_time(svm_d1, "SVM", name1)
plot_f1_vs_time(rf_d2, "Random Forest", name2)
plot_f1_vs_time(svm_d2, "SVM", name2)

# txt com as melhores configurações de hiperparâmetros
txt_path = os.path.join(OUT_DIR, "melhores_configs.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("Melhores Configurações de Hiperparâmetros\n\n")
    for label, info, df in [
        (f"RandomForest - {name1}", info_rf_d1, rf_d1),
        (f"SVM - {name1}", info_svm_d1, svm_d1),
        (f"RandomForest - {name2}", info_rf_d2, rf_d2),
        (f"SVM - {name2}", info_svm_d2, svm_d2),
    ]:
        best = df.iloc[df['f1'].idxmax()]
        f.write(f"{label}\n")
        f.write(f"Melhor F1: {best['f1']:.4f}\n")
        f.write(f"Recall: {best['recall']:.4f}\n")
        f.write(f"Tempo de treino: {best['mean_fit_time']:.3f}s\n")
        f.write(f"Tempo de predição: {best['mean_predict_time']:.6f}s\n")
        f.write(f"Tamanho do modelo: {best['model_size_bytes']:,} bytes\n")
        f.write(f"Tempo total de busca: {info['total_search_time']:.2f}s\n")
        f.write(f"Parâmetros: {json.dumps(info['best_params'], ensure_ascii=False, indent=2)}\n")
        f.write("-" * 60 + "\n\n")

print(f"Resultados salvos em: {OUT_DIR}")
