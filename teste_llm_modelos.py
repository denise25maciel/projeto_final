from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any

import numpy as np
from dotenv import load_dotenv

from config import MODELO_LLM_PADRAO
from utils_json import extrair_json


# ---------------------------------------------------------------------------
# Estatísticas da série temporal (modo resumido para o LLM)
# ---------------------------------------------------------------------------

def _calcular_stats_segmento(
    arr_seg: np.ndarray,
    offset: int,
    datas_serie: list[str] | None,
) -> dict:
    """Estatísticas de um segmento da série, com posições absolutas (1-based)."""
    n = len(arr_seg)
    if n == 0:
        return {}

    media = float(np.mean(arr_seg))
    desvio = float(np.std(arr_seg, ddof=1)) if n > 1 else 0.0
    cv = round(desvio / media, 4) if media != 0 else 0.0

    def _ac(x: np.ndarray, lag: int) -> float:
        if len(x) <= lag:
            return 0.0
        c = np.corrcoef(x[:-lag], x[lag:])[0, 1]
        return round(float(c) if np.isfinite(c) else 0.0, 4)

    def _info(local_idx: int) -> dict:
        abs_idx = offset + local_idx
        entry: dict = {"posicao": abs_idx + 1, "valor": int(arr_seg[local_idx])}
        if datas_serie and abs_idx < len(datas_serie):
            entry["data"] = datas_serie[abs_idx]
        return entry

    picos = [
        _info(i)
        for i in range(1, n - 1)
        if arr_seg[i] > arr_seg[i - 1] and arr_seg[i] > arr_seg[i + 1]
    ]

    if desvio > 0:
        zscores = np.abs((arr_seg - media) / desvio)
        idx_z = int(np.argmax(zscores))
        zscore_max_info: dict = {**_info(idx_z), "zscore": round(float(zscores[idx_z]), 4)}
    else:
        zscore_max_info = {"posicao": None, "valor": None, "zscore": 0.0}

    if n > 1:
        diffs = np.abs(np.diff(arr_seg))
        idx_r = int(np.argmax(diffs))
        ruptura_seg: dict = {
            "entre_posicoes": [offset + idx_r + 1, offset + idx_r + 2],
            "diferenca": round(float(diffs[idx_r]), 4),
            "media_antes": round(float(np.mean(arr_seg[: idx_r + 1])), 4),
            "media_depois": round(float(np.mean(arr_seg[idx_r + 1:])), 4),
        }
    else:
        ruptura_seg = None

    return {
        "posicoes": [offset + 1, offset + n],
        "n_pontos": n,
        "media": round(media, 4),
        "desvio_padrao": round(desvio, 4),
        "cv": cv,
        "minimo": int(np.min(arr_seg)),
        "maximo": int(np.max(arr_seg)),
        "mediana": float(np.median(arr_seg)),
        "autocorrelacao_lag1": _ac(arr_seg, 1),
        "picos": picos,
        "zscore_maximo": zscore_max_info,
        "ponto_ruptura": ruptura_seg,
    }


def calcular_estatisticas_serie(
    valores_serie: list[int],
    datas_serie: list[str] | None = None,
) -> dict:
    """
    Computa estatísticas globais + por quartil temporal da série.
    Retorna {"global": {...}, "quartil_1": {...}, ..., "quartil_4": {...}}.
    Usado quando modo_entrada_llm == 'estatisticas'.
    """
    arr = np.array(valores_serie, dtype=float)
    n = len(arr)

    media = float(np.mean(arr))
    desvio = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    cv = round(desvio / media, 4) if media != 0 else 0.0

    def _autocorr(x: np.ndarray, lag: int) -> float:
        if len(x) <= lag:
            return 0.0
        c = np.corrcoef(x[:-lag], x[lag:])[0, 1]
        return round(float(c) if np.isfinite(c) else 0.0, 4)

    def _info_ponto(idx: int) -> dict:
        entry: dict = {"posicao": idx + 1, "valor": int(arr[idx])}
        if datas_serie and idx < len(datas_serie):
            entry["data"] = datas_serie[idx]
        return entry

    picos_globais = [
        _info_ponto(i)
        for i in range(1, n - 1)
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]
    ]

    desvio_movel_3 = [
        round(float(np.std(arr[i: i + 3], ddof=0)), 4) for i in range(n - 2)
    ]

    if desvio > 0:
        zscores = np.abs((arr - media) / desvio)
        idx_z = int(np.argmax(zscores))
        zscore_max_global = {**_info_ponto(idx_z), "zscore": round(float(zscores[idx_z]), 4)}
    else:
        zscore_max_global = {"posicao": None, "valor": None, "zscore": 0.0}

    if n > 1:
        diffs = np.abs(np.diff(arr))
        idx_r = int(np.argmax(diffs))
        ruptura_global: dict = {
            "entre_posicoes": [idx_r + 1, idx_r + 2],
            "diferenca": round(float(diffs[idx_r]), 4),
            "media_antes": round(float(np.mean(arr[: idx_r + 1])), 4),
            "media_depois": round(float(np.mean(arr[idx_r + 1:])), 4),
        }
        if datas_serie:
            ruptura_global["datas"] = [
                datas_serie[idx_r] if idx_r < len(datas_serie) else None,
                datas_serie[idx_r + 1] if idx_r + 1 < len(datas_serie) else None,
            ]
    else:
        ruptura_global = None

    # Divisão em 4 quartis temporais (último quartil absorve o restante)
    tamanho_q = n // 4
    limites = [q * tamanho_q for q in range(4)] + [n]
    quartis = [
        _calcular_stats_segmento(arr[limites[q]: limites[q + 1] if q < 3 else n], limites[q], datas_serie)
        for q in range(4)
    ]

    return {
        "global": {
            "n_pontos": n,
            "media": round(media, 4),
            "desvio_padrao": round(desvio, 4),
            "cv": cv,
            "minimo": int(np.min(arr)),
            "maximo": int(np.max(arr)),
            "mediana": float(np.median(arr)),
            "autocorrelacao_lag1": _autocorr(arr, 1),
            "autocorrelacao_lag7": _autocorr(arr, 7),
            "picos": picos_globais,
            "desvio_padrao_movel_3": desvio_movel_3,
            "zscore_maximo": zscore_max_global,
            "ponto_ruptura": ruptura_global,
        },
        "quartil_1": quartis[0],
        "quartil_2": quartis[1],
        "quartil_3": quartis[2],
        "quartil_4": quartis[3],
    }

# ---------------------------------------------------------------------------
# Feature importance por tipo de algoritmo
# ---------------------------------------------------------------------------

def extrair_feature_importances(
    modelo,
    feature_cols: list[str],
    tipo_algoritmo: str,
    top_n: int = 10,
) -> list[dict]:
    """
    Extrai as top_n features mais importantes por tipo de algoritmo.

    - Random Forest : feature_importances_ (Gini/MDI)
    """
    if tipo_algoritmo == "random_forest":
        importances = modelo.feature_importances_
    else:
        return []

    pares = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)[:top_n]
    return [
        {"coluna": col, "importancia_global": round(float(imp), 4)}
        for col, imp in pares
    ]


# ---------------------------------------------------------------------------
# Top regras / caminho de decisão (apenas modelos de árvore)
# ---------------------------------------------------------------------------

def extrair_top_regras(
    modelo,
    x_linha,
    feature_cols: list[str],
    tipo_algoritmo: str,
    top_n: int = 10,
) -> list[dict]:
    """Para Random Forest: caminhos de decisão em todas as árvores."""
    if tipo_algoritmo == "random_forest":
        return _top_regras_rf(modelo, x_linha, feature_cols, top_n)
    return []


def _top_regras_rf(modelo_rf, x_linha, feature_cols, top_n):
    contador: Counter = Counter()
    for arvore in modelo_rf.estimators_:
        tree = arvore.tree_
        caminho = arvore.decision_path(x_linha[feature_cols])
        for node_id in caminho.indices:
            feature_idx = tree.feature[node_id]
            if feature_idx >= 0:
                contador[feature_cols[feature_idx]] += 1

    total_arvores = len(modelo_rf.estimators_)
    return [
        {
            "coluna": col,
            "valor_observado": float(x_linha[col].iloc[0]),
            "frequencia_nas_arvores": freq,
            "percentual_arvores": round(freq / total_arvores, 4),
        }
        for col, freq in contador.most_common(top_n)
    ]


# ---------------------------------------------------------------------------
# Probabilidades / pseudo-probabilidades por tipo de algoritmo
# ---------------------------------------------------------------------------

def _predict_proba_generico(
    modelo,
    X,
    tipo_algoritmo: str,
    classes: list[str],
    label_encoder=None,
) -> np.ndarray:
    """Retorna matriz (n, n_classes) de probabilidades."""
    return modelo.predict_proba(X)


def _predict_generico(
    modelo,
    X,
    tipo_algoritmo: str,
    label_encoder=None,
) -> np.ndarray:
    return modelo.predict(X)


# ---------------------------------------------------------------------------
# Seleção de exemplo representativo por classe
# ---------------------------------------------------------------------------

def selecionar_exemplo_classe(
    df_matriz,
    idx_train: list,
    feature_cols: list[str],
    modelo,
    classe: str,
    tipo_algoritmo: str,
    classes: list[str],
    label_encoder=None,
) -> dict:
    """
    Seleciona o exemplo de treino mais representativo da classe especificada:
    aquele com maior probabilidade prevista pelo modelo para essa classe.
    """
    classe_idx = classes.index(classe)
    df_train = df_matriz.loc[idx_train].copy()
    df_classe = df_train[df_train["classe_real"] == classe]

    if df_classe.empty:
        return {}

    probas = _predict_proba_generico(
        modelo, df_classe[feature_cols], tipo_algoritmo, classes, label_encoder
    )
    melhor_idx = df_classe.index[probas[:, classe_idx].argmax()]

    x_linha = df_matriz.loc[[melhor_idx], feature_cols]
    valores_serie = [int(x_linha[col].iloc[0]) for col in feature_cols]
    datas_serie = [col.replace("dia_", "").replace("_", "-") for col in feature_cols]

    return {
        "classe": classe,
        "valores_serie": valores_serie,
        "datas_serie": datas_serie,
    }


# ---------------------------------------------------------------------------
# Prompt genérico
# ---------------------------------------------------------------------------

def _bloco_serie_bruta(valores_serie: list[int], datas_serie: list[str] | None) -> str:
    """Retorna o trecho de série bruta a ser anexado ao bloco_serie dos modos estatísticos."""
    if datas_serie is not None and len(datas_serie) == len(valores_serie):
        serie = json.dumps(
            [{"data": str(d), "valor": v} for d, v in zip(datas_serie, valores_serie)],
            ensure_ascii=False,
        )
    else:
        serie = json.dumps(valores_serie, ensure_ascii=False)
    return f"\n\nSérie bruta (valores diários completos, para referência):\n{serie}"


def montar_prompt_llm(
    id_serie: str,
    valores_serie: list[int],
    nome_algoritmo: str,
    classe_predita: str,
    probabilidades: dict,
    datas_serie: list | None = None,
    top_regras: list[dict] | None = None,
    feature_importances: list[dict] | None = None,
    incluir_feature_importance: bool = True,
    modo_entrada_llm: str = "bruto",
    incluir_serie_bruta: bool = False,
) -> str:
    # --- bloco_serie ---
    estatisticas: dict | None = None
    if modo_entrada_llm == "estatisticas":
        estatisticas = calcular_estatisticas_serie(valores_serie, datas_serie)
        q1 = estatisticas["quartil_1"]
        q2 = estatisticas["quartil_2"]
        q3 = estatisticas["quartil_3"]
        q4 = estatisticas["quartil_4"]
        _aviso_bruto_est = "" if incluir_serie_bruta else " (dados brutos não fornecidos)"
        bloco_serie = (
            f"Série temporal dividida em 4 quartis temporais{_aviso_bruto_est}:\n\n"
            "Estatísticas globais:\n"
            + json.dumps(estatisticas["global"], ensure_ascii=False, indent=2)
            + f"\n\nQuartil 1 — posições {q1.get('posicoes', [])}:\n"
            + json.dumps(q1, ensure_ascii=False, indent=2)
            + f"\n\nQuartil 2 — posições {q2.get('posicoes', [])}:\n"
            + json.dumps(q2, ensure_ascii=False, indent=2)
            + f"\n\nQuartil 3 — posições {q3.get('posicoes', [])}:\n"
            + json.dumps(q3, ensure_ascii=False, indent=2)
            + f"\n\nQuartil 4 — posições {q4.get('posicoes', [])}:\n"
            + json.dumps(q4, ensure_ascii=False, indent=2)
            + "\n\nNota: O campo 'posicao' nas estatísticas corresponde ao índice 1-based "
            "que deve ser usado no campo \"dia\" da resposta."
        )
        if incluir_serie_bruta:
            bloco_serie += _bloco_serie_bruta(valores_serie, datas_serie)
    elif modo_entrada_llm == "global":
        estatisticas = calcular_estatisticas_serie(valores_serie, datas_serie)
        _aviso_bruto = "" if incluir_serie_bruta else " (dados brutos e divisão por quartis não fornecidos)"
        bloco_serie = (
            f"Série temporal — apenas estatísticas globais{_aviso_bruto}:\n\n"
            + json.dumps(estatisticas["global"], ensure_ascii=False, indent=2)
            + "\n\nNota: O campo 'posicao' nas estatísticas corresponde ao índice 1-based "
            "que deve ser usado no campo \"dia\" da resposta."
        )
        if incluir_serie_bruta:
            bloco_serie += _bloco_serie_bruta(valores_serie, datas_serie)
    elif datas_serie is not None and len(datas_serie) == len(valores_serie):
        serie_formatada = json.dumps(
            [{"data": str(d), "valor": v} for d, v in zip(datas_serie, valores_serie)],
            ensure_ascii=False,
        )
        bloco_serie = f"Série temporal (data + valor de acesso):\n{serie_formatada}"
    else:
        bloco_serie = f"Valores completos, em ordem temporal:\n{json.dumps(valores_serie, ensure_ascii=False)}"

    # --- bloco_tarefa e bloco_formato_json (condicionais por modo) ---
    _ponto_exemplo = (
        '    {\n'
        '      "dia": 1,\n'
        '      "valor": 10,\n'
        '      "importancia": 0.90,\n'
        '      "sentido": "evidencia",\n'
        '      "motivo_associado": "pico_extremo",\n'
        '      "justificativa": "Este ponto representa o maior valor da série..."\n'
        '    }'
    )

    if modo_entrada_llm == "global" and estatisticas is not None:
        bloco_tarefa = (
            "Tarefa:\n"
            "1. Analise as estatísticas globais da série temporal.\n"
            "2. Identifique os pontos mais significativos com base nessas estatísticas "
            "(picos, outliers, ponto de ruptura, valor mínimo/máximo, etc.).\n"
            "3. Para cada ponto selecionado, justifique sua relevância estatística.\n\n"
            "Regras para os pontos:\n"
            "- Use o campo \"dia\" com o valor do campo \"posicao\" nas estatísticas globais.\n"
            "- O campo \"valor\" deve ser exatamente o valor informado para aquele ponto.\n"
            "- O campo \"importancia\" deve ser um número entre 0 e 1 (relevância estatística).\n"
            "- O campo \"sentido\" deve ser \"evidencia\".\n"
            "- Escolha de 3 a 20 pontos emergindo das estatísticas globais."
        )
        bloco_formato_json = (
            '{\n'
            f'  "id_serie": "{id_serie}",\n'
            '  "analise_global_da_serie": "Síntese baseada nas estatísticas globais.",\n'
            '  "motivos_identificados_pelo_llm": ["padrão_1", "padrão_2"],\n'
            '  "llm_pontos": [\n'
            + _ponto_exemplo + '\n'
            '  ]\n'
            '}'
        )
    elif modo_entrada_llm == "estatisticas" and estatisticas is not None:
        q1_pos = estatisticas["quartil_1"].get("posicoes", ["?", "?"])
        q2_pos = estatisticas["quartil_2"].get("posicoes", ["?", "?"])
        q3_pos = estatisticas["quartil_3"].get("posicoes", ["?", "?"])
        q4_pos = estatisticas["quartil_4"].get("posicoes", ["?", "?"])

        bloco_tarefa = (
            "Tarefa — siga obrigatoriamente esta sequência:\n\n"
            f"Passo 1 — Análise do Quartil 1 (posições {q1_pos[0]} a {q1_pos[-1]}): "
            "descreva o comportamento estatístico desse segmento "
            "(nível médio, variação, picos, outliers e regularidade).\n"
            f"Passo 2 — Análise do Quartil 2 (posições {q2_pos[0]} a {q2_pos[-1]}): idem.\n"
            f"Passo 3 — Análise do Quartil 3 (posições {q3_pos[0]} a {q3_pos[-1]}): idem.\n"
            f"Passo 4 — Análise do Quartil 4 (posições {q4_pos[0]} a {q4_pos[-1]}): idem.\n"
            "Passo 5 — Síntese global: integre as quatro análises e selecione os pontos "
            "estatisticamente mais relevantes da série.\n\n"
            "Regras para os pontos:\n"
            "- Use o campo \"dia\" com o valor do campo \"posicao\" nas estatísticas dos quartis.\n"
            "- O campo \"valor\" deve ser exatamente o valor informado para aquele ponto.\n"
            "- O campo \"importancia\" deve ser um número entre 0 e 1 (relevância estatística).\n"
            "- O campo \"sentido\" deve ser \"evidencia\".\n"
            "- Escolha de 3 a 20 pontos emergindo da síntese global, "
            "não distribua mecanicamente pontos entre os quartis."
        )
        bloco_formato_json = (
            '{\n'
            f'  "id_serie": "{id_serie}",\n'
            '  "analise_quartil_1": "Análise do Quartil 1.",\n'
            '  "analise_quartil_2": "Análise do Quartil 2.",\n'
            '  "analise_quartil_3": "Análise do Quartil 3.",\n'
            '  "analise_quartil_4": "Análise do Quartil 4.",\n'
            '  "analise_global_da_serie": "Síntese dos 4 quartis (Passo 5).",\n'
            '  "motivos_identificados_pelo_llm": ["padrão_1", "padrão_2"],\n'
            '  "llm_pontos": [\n'
            + _ponto_exemplo + '\n'
            '  ]\n'
            '}'
        )
    else:
        bloco_tarefa = (
            "Tarefa:\n"
            "1. Leia a série temporal completa e faça uma análise global do seu comportamento.\n"
            "2. Identifique os pontos estatisticamente mais significativos: "
            "picos extremos, vales, rupturas abruptas, outliers ou regiões de alta variabilidade.\n"
            "3. Para cada ponto selecionado, justifique sua relevância com base no comportamento observado.\n\n"
            "Regras para os pontos:\n"
            "- Use o campo \"dia\" como posição da série começando em 1.\n"
            "- O campo \"valor\" deve ser exatamente o valor observado naquele dia.\n"
            "- O campo \"importancia\" deve ser um número entre 0 e 1 (relevância estatística).\n"
            "- O campo \"sentido\" deve ser \"evidencia\".\n"
            "- O campo \"motivo_associado\" deve descrever o padrão observado "
            "(ex: \"pico_extremo\", \"vale_outlier\", \"ruptura_abrupta\").\n"
            "- Escolha de 3 a 20 pontos emergindo da análise global, "
            "não de uma varredura isolada ponto a ponto."
        )
        bloco_formato_json = (
            '{\n'
            f'  "id_serie": "{id_serie}",\n'
            '  "analise_global_da_serie": "Análise global da série.",\n'
            '  "motivos_identificados_pelo_llm": ["padrão_1", "padrão_2"],\n'
            '  "llm_pontos": [\n'
            + _ponto_exemplo + '\n'
            '  ]\n'
            '}'
        )

    return f"""
Você é um analista de séries temporais educacionais.

{bloco_tarefa}

Série temporal analisada:
ID: {id_serie}
{bloco_serie}

Responda obrigatoriamente em JSON válido, exatamente neste formato:

{bloco_formato_json}
"""


# ---------------------------------------------------------------------------
# Validação da resposta LLM (igual ao módulo RF)
# ---------------------------------------------------------------------------

def _validar_resposta_llm(
    resposta: dict[str, Any],
    id_serie: str,
    valores_serie: list[int],
) -> dict[str, Any]:
    resposta = dict(resposta)
    resposta.setdefault("id_serie", id_serie)
    resposta.setdefault("analise_quartil_1", "")
    resposta.setdefault("analise_quartil_2", "")
    resposta.setdefault("analise_quartil_3", "")
    resposta.setdefault("analise_quartil_4", "")
    resposta.setdefault("analise_global_da_serie", "")
    resposta.setdefault("motivos_identificados_pelo_llm", [])
    resposta.setdefault("llm_pontos", [])

    motivos = resposta.get("motivos_identificados_pelo_llm", [])
    if not isinstance(motivos, list):
        motivos = []
    resposta["motivos_identificados_pelo_llm"] = [
        str(m) for m in motivos if isinstance(m, str) and m.strip()
    ]

    pontos_normalizados: list[dict[str, Any]] = []
    for item in resposta.get("llm_pontos", []):
        if not isinstance(item, dict):
            continue
        try:
            dia = int(item.get("dia"))
        except Exception:
            continue
        if dia < 1 or dia > len(valores_serie):
            continue

        try:
            importancia = float(item.get("importancia", 0))
        except Exception:
            importancia = 0.0
        importancia = max(0.0, min(1.0, importancia))

        sentido = str(item.get("sentido", "evidencia")).strip().lower()
        if sentido not in {"favorece", "reduz", "evidencia"}:
            sentido = "evidencia"

        pontos_normalizados.append({
            "dia": dia,
            "valor": int(valores_serie[dia - 1]),
            "importancia": round(importancia, 4),
            "sentido": sentido,
            "motivo_associado": str(item.get("motivo_associado", "")).strip(),
            "justificativa": str(item.get("justificativa", "")).strip(),
            "metodo": "LLM",
        })

    pontos_normalizados = sorted(
        pontos_normalizados, key=lambda p: float(p.get("importancia", 0)), reverse=True
    )[:10]
    for i, ponto in enumerate(pontos_normalizados, start=1):
        ponto["ranking"] = i

    resposta["llm_pontos"] = pontos_normalizados
    return resposta


# ---------------------------------------------------------------------------
# Execução da chamada LLM
# ---------------------------------------------------------------------------

def executar_teste_llm(
    id_serie: str,
    valores_serie: list[int],
    modelo,
    tipo_algoritmo: str,
    nome_algoritmo: str,
    classes: list[str],
    df_matriz,
    idx_train: list,
    feature_cols: list[str],
    incluir_feature_importance: bool = True,
    incluir_top_regras: bool = True,
    label_encoder=None,
    datas_serie: list | None = None,
    modelo_llm: str = MODELO_LLM_PADRAO,
    modo_entrada_llm: str = "bruto",
    incluir_serie_bruta: bool = False,
) -> dict:
    """
    Executa a chamada ao LLM para uma série de teste.

    incluir_feature_importance controla se o feature importance do algoritmo
    será incluído no prompt enviado ao LLM.
    """
    import pandas as pd

    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY não encontrada. Crie um arquivo .env com OPENAI_API_KEY=sua_chave.")

    x_linha = df_matriz.loc[[id_serie], feature_cols] if id_serie in df_matriz.index else None
    if x_linha is None:
        idx_pos = df_matriz["id_serie"].tolist().index(id_serie)
        x_linha = df_matriz.iloc[[idx_pos]][feature_cols]

    # Predição e probabilidades
    classe_predita = _predict_generico(modelo, x_linha, tipo_algoritmo, label_encoder)[0]
    probas = _predict_proba_generico(modelo, x_linha, tipo_algoritmo, classes, label_encoder)[0]
    probabilidades = {c: round(float(p), 4) for c, p in zip(classes, probas)}

    # Feature importance (apenas se o usuário solicitou)
    feature_importances = (
        extrair_feature_importances(modelo, feature_cols, tipo_algoritmo, top_n=10)
        if incluir_feature_importance
        else None
    )

    # Top regras / caminho de decisão (apenas para árvores)
    top_regras = (
        extrair_top_regras(modelo, x_linha, feature_cols, tipo_algoritmo, top_n=10)
        if incluir_top_regras
        else None
    )

    prompt = montar_prompt_llm(
        id_serie=id_serie,
        valores_serie=valores_serie,
        nome_algoritmo=nome_algoritmo,
        classe_predita=classe_predita,
        probabilidades=probabilidades,
        datas_serie=datas_serie,
        top_regras=top_regras,
        feature_importances=feature_importances,
        incluir_feature_importance=incluir_feature_importance,
        modo_entrada_llm=modo_entrada_llm,
        incluir_serie_bruta=incluir_serie_bruta,
    )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("A biblioteca openai não está instalada. Execute: pip install openai") from exc

    client = OpenAI(api_key=api_key)
    resposta = client.responses.create(model=modelo_llm, input=prompt, temperature=0)

    resposta_json = extrair_json(resposta.output_text)
    resultado = _validar_resposta_llm(resposta_json, id_serie=id_serie, valores_serie=valores_serie)

    # Adiciona metadados do algoritmo ao resultado
    resultado["tipo_algoritmo"] = tipo_algoritmo
    resultado["nome_algoritmo"] = nome_algoritmo
    resultado["feature_importance_incluida"] = incluir_feature_importance
    resultado["classe_predita_pelo_algoritmo"] = classe_predita
    resultado["probabilidades_algoritmo"] = probabilidades
    resultado["_prompt"] = prompt

    return resultado
