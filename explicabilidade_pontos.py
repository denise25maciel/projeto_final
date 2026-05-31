from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def _extrair_data_coluna_dia(coluna: str) -> str:
    """Converte dia_2026_02_01 em 2026-02-01, quando possível."""
    match = re.match(r"dia_(\d{4})_(\d{2})_(\d{2})$", str(coluna))
    if not match:
        return str(coluna)
    ano, mes, dia = match.groups()
    return f"{ano}-{mes}-{dia}"


def _normalizar_ranking(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []

    df = df.copy()
    df["abs_contribuicao"] = df["contribuicao"].abs()
    df = df.sort_values("abs_contribuicao", ascending=False).reset_index(drop=True)
    df["ranking"] = range(1, len(df) + 1)
    return df.to_dict(orient="records")


def calcular_pontos_shap(
    modelo_rf: Any,
    x_linha: pd.DataFrame,
    feature_cols: list[str],
    classe_predita: str,
) -> list[dict[str, Any]]:
    """
    Calcula a contribuição SHAP de cada ponto/dia da série temporal.

    Retorna uma lista de dicionários com dia, data, valor e contribuição. A lista
    já vem ordenada pelo maior impacto absoluto.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "A biblioteca shap não está instalada. Execute: pip install shap"
        ) from exc

    classe_idx = list(modelo_rf.classes_).index(classe_predita)
    explainer = shap.TreeExplainer(modelo_rf)
    shap_values = explainer.shap_values(x_linha)

    if isinstance(shap_values, list):
        valores_shap = np.asarray(shap_values[classe_idx])[0]
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            valores_shap = arr[0, :, classe_idx]
        elif arr.ndim == 2:
            valores_shap = arr[0]
        else:
            valores_shap = arr.reshape(-1)

    registros: list[dict[str, Any]] = []
    valores_serie = x_linha.iloc[0].to_dict()

    for i, coluna in enumerate(feature_cols):
        registros.append({
            "dia": i + 1,
            "data": _extrair_data_coluna_dia(coluna),
            "coluna": coluna,
            "valor": float(valores_serie[coluna]),
            "contribuicao": float(valores_shap[i]),
            "metodo": "SHAP",
        })

    return _normalizar_ranking(pd.DataFrame(registros))


def calcular_pontos_lime(
    modelo_rf: Any,
    x_treino: pd.DataFrame,
    x_linha: pd.DataFrame,
    feature_cols: list[str],
    classe_predita: str,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """
    Calcula a contribuição LIME de cada ponto/dia da série temporal.

    Usa discretize_continuous=False para manter uma explicação por coluna dia_*,
    o que facilita comparar diretamente com o SHAP no gráfico da série.
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as exc:
        raise ImportError(
            "A biblioteca lime não está instalada. Execute: pip install lime"
        ) from exc

    classe_idx = list(modelo_rf.classes_).index(classe_predita)

    explainer = LimeTabularExplainer(
        training_data=x_treino[feature_cols].to_numpy(),
        feature_names=feature_cols,
        class_names=[str(c) for c in modelo_rf.classes_],
        mode="classification",
        discretize_continuous=False,
        random_state=random_state,
    )

    explicacao = explainer.explain_instance(
        data_row=x_linha[feature_cols].iloc[0].to_numpy(),
        predict_fn=modelo_rf.predict_proba,
        num_features=len(feature_cols),
        labels=[classe_idx],
    )

    pesos_por_indice = dict(explicacao.as_map().get(classe_idx, []))
    valores_serie = x_linha.iloc[0].to_dict()

    registros: list[dict[str, Any]] = []
    for i, coluna in enumerate(feature_cols):
        registros.append({
            "dia": i + 1,
            "data": _extrair_data_coluna_dia(coluna),
            "coluna": coluna,
            "valor": float(valores_serie[coluna]),
            "contribuicao": float(pesos_por_indice.get(i, 0.0)),
            "metodo": "LIME",
        })

    return _normalizar_ranking(pd.DataFrame(registros))


def selecionar_top_pontos(pontos: list[dict[str, Any]], top_n: int = 10) -> list[dict[str, Any]]:
    """Retorna apenas os N pontos de maior contribuição absoluta."""
    return sorted(
        pontos,
        key=lambda item: abs(float(item.get("contribuicao", 0))),
        reverse=True,
    )[:top_n]
