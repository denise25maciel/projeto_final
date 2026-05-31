from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from config import (
    ARQUIVO_BASE,
    ARQUIVO_MATRIZ_RF,
    ARQUIVO_MODELO_RF,
)
from features_series import montar_matriz_series_temporais

CONFIGS_ALGORITMOS: dict[str, dict] = {
    "random_forest": {
        "nome": "Random Forest",
        "arquivo_modelo": ARQUIVO_MODELO_RF,
    },
}


def _instanciar_modelo(chave: str, random_state: int = 42):
    if chave == "random_forest":
        return RandomForestClassifier(
            n_estimators=120,
            max_depth=5,
            random_state=random_state,
            class_weight="balanced",
        )
    raise ValueError(f"Algoritmo desconhecido: {chave}")


def treinar_todos_os_modelos(
    df: pd.DataFrame | None = None,
    random_state: int = 42,
) -> tuple[dict, pd.DataFrame]:
    """
    Treina o Random Forest.
    Retorna um dicionário de pacotes por algoritmo e a matriz de features.
    """
    if df is None:
        df = pd.read_csv(ARQUIVO_BASE)

    df_matriz, feature_cols = montar_matriz_series_temporais(df)

    X = df_matriz[feature_cols]
    y = df_matriz["classe_real"]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        df_matriz.index,
        test_size=0.35,
        random_state=random_state,
        stratify=y,
    )

    resultados: dict[str, dict] = {}

    for chave, info in CONFIGS_ALGORITMOS.items():
        modelo = _instanciar_modelo(chave, random_state)
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)

        acc      = accuracy_score(y_test, pred)
        f1       = f1_score(y_test, pred, average="weighted", zero_division=0)
        precisao = precision_score(y_test, pred, average="macro", zero_division=0)
        recall   = recall_score(y_test, pred, average="macro", zero_division=0)
        report   = classification_report(y_test, pred, output_dict=True, zero_division=0)

        pacote = {
            "modelo":               modelo,
            "feature_cols":         feature_cols,
            "idx_train":            list(idx_train),
            "idx_test":             list(idx_test),
            "accuracy":             acc,
            "f1_weighted":          f1,
            "precisao_macro":       precisao,
            "recall_macro":         recall,
            "classification_report": report,
            "tipo_algoritmo":       chave,
            "nome_algoritmo":       info["nome"],
            "classes":              list(modelo.classes_),
            "label_encoder":        None,
        }

        joblib.dump(pacote, info["arquivo_modelo"])
        resultados[chave] = pacote

    df_matriz.to_csv(ARQUIVO_MATRIZ_RF, index=False, encoding="utf-8-sig")
    return resultados, df_matriz


def exibir_tabela_metricas(resultados: dict) -> None:
    c1, c2, c3, c4, c5 = 24, 12, 13, 18, 15
    total = c1 + c2 + c3 + c4 + c5 + 4
    sep = "=" * total
    print(f"\n{sep}")
    print(f"{'MÉTRICAS COMPARATIVAS DOS ALGORITMOS':^{total}}")
    print(sep)
    print(
        f"{'Algoritmo':<{c1}} {'Acurácia':>{c2}} {'F1 Weighted':>{c3}}"
        f" {'Precisão (macro)':>{c4}} {'Recall (macro)':>{c5}}"
    )
    print("-" * total)
    for pacote in resultados.values():
        print(
            f"{pacote['nome_algoritmo']:<{c1}}"
            f" {pacote['accuracy']:>{c2}.4f}"
            f" {pacote['f1_weighted']:>{c3}.4f}"
            f" {pacote['precisao_macro']:>{c4}.4f}"
            f" {pacote['recall_macro']:>{c5}.4f}"
        )
    print(sep)


if __name__ == "__main__":
    resultados, _ = treinar_todos_os_modelos()
    exibir_tabela_metricas(resultados)
