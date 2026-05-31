from __future__ import annotations

import pandas as pd


def obter_colunas_serie(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("dia_")]


def montar_matriz_series_temporais(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    colunas_serie = obter_colunas_serie(df)

    if not colunas_serie:
        raise ValueError("Nenhuma coluna dia_* foi encontrada na base.")

    df_matriz = df[["id_serie", "classe_real"] + colunas_serie].copy()

    for col in colunas_serie:
        df_matriz[col] = pd.to_numeric(df_matriz[col], errors="coerce")

    if df_matriz[colunas_serie].isna().any().any():
        colunas_invalidas = df_matriz[colunas_serie].columns[df_matriz[colunas_serie].isna().any()].tolist()
        raise ValueError(f"Há valores ausentes ou não numéricos nas colunas: {colunas_invalidas}")

    return df_matriz, colunas_serie
