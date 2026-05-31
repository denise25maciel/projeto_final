from __future__ import annotations

import json

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from config import ARQUIVO_BASE, ARQUIVO_EVENTOS, ARQUIVO_RELATORIO_LLM, ARQUIVO_RESULTADO_COMPARACAO
from explicabilidade_pontos import (
    calcular_pontos_lime,
    calcular_pontos_shap,
    selecionar_top_pontos,
)
from features_series import obter_colunas_serie
from gerar_base import gerar_base_sintetica
from teste_llm_modelos import executar_teste_llm, selecionar_exemplo_classe
from treinar_modelos import treinar_todos_os_modelos


def comparar_motivos(motivos_especialista: list[str], motivos_llm: list[str]) -> dict:
    set_esp = set(motivos_especialista)
    set_llm = set(motivos_llm)
    intersecao = set_esp & set_llm
    uniao = set_esp | set_llm
    score = len(intersecao) / len(uniao) if uniao else 0.0

    if score == 1.0:
        alinhamento = "sim"
    elif score > 0:
        alinhamento = "parcial"
    else:
        alinhamento = "nao"

    return {
        "alinhamento_llm_especialista": alinhamento,
        "score_alinhamento_motivos": round(score, 4),
        "motivos_em_comum": sorted(intersecao),
        "motivos_ausentes_no_llm": sorted(set_esp - set_llm),
        "motivos_extras_do_llm": sorted(set_llm - set_esp),
    }


def _json_seguro(objeto) -> str:
    return json.dumps(objeto, ensure_ascii=False)


def preparar_pontos_llm_para_visualizacao(
    pontos_llm: list[dict],
    valores_serie: list[int],
    colunas_serie: list[str],
) -> list[dict]:
    registros: list[dict] = []

    if not isinstance(pontos_llm, list):
        return registros

    for item in pontos_llm:
        if not isinstance(item, dict):
            continue
        try:
            dia = int(item.get("dia"))
        except Exception:
            continue
        if dia < 1 or dia > len(valores_serie):
            continue

        try:
            importancia = float(item.get("importancia", 0.0))
        except Exception:
            importancia = 0.0
        importancia = max(0.0, min(1.0, importancia))

        sentido = str(item.get("sentido", "evidencia")).strip().lower()
        sinal = -1.0 if sentido == "reduz" else 1.0
        registros.append({
            "dia": dia,
            "data": colunas_serie[dia - 1].replace("dia_", "").replace("_", "-"),
            "coluna": colunas_serie[dia - 1],
            "valor": float(valores_serie[dia - 1]),
            "contribuicao": round(sinal * importancia, 4),
            "importancia": importancia,
            "sentido": sentido,
            "motivo_associado": str(item.get("motivo_associado", "")),
            "justificativa": str(item.get("justificativa", "")),
            "metodo": "LLM",
        })

    registros = sorted(registros, key=lambda x: float(x.get("importancia", 0.0)), reverse=True)
    for i, registro in enumerate(registros, start=1):
        registro["ranking"] = i
    return registros


def _pred_classe(modelo, X, tipo_algoritmo: str, label_encoder, classes: list[str]) -> str:
    return str(modelo.predict(X)[0])


def _pred_probas(modelo, X, tipo_algoritmo: str, classes: list[str], label_encoder) -> dict:
    probas_arr = modelo.predict_proba(X)[0]
    return {str(c): round(float(p), 4) for c, p in zip(classes, probas_arr)}


def _campos_llm_vazios(prefixo: str) -> dict:
    return {
        f"{prefixo}_pontos":          "[]",
        f"{prefixo}_top_20":          "[]",
        f"{prefixo}_previsto":         "",
        f"{prefixo}_confianca":        "",
        f"{prefixo}_explicacao":       "",
        f"{prefixo}_analise":          "",
        f"{prefixo}_motivos":          "[]",
        f"{prefixo}_alinhamento":      "",
        f"{prefixo}_score":            "",
        f"{prefixo}_motivos_em_comum": "[]",
        f"{prefixo}_motivos_ausentes": "[]",
        f"{prefixo}_motivos_extras":   "[]",
        f"{prefixo}_erro":             "",
    }


def _rodar_llm(
    modo: str,
    row_original,
    valores_serie: list[int],
    modelo,
    tipo_algoritmo: str,
    nome_algoritmo: str,
    classes: list[str],
    df_matriz,
    idx_train: list,
    feature_cols: list[str],
    incluir_feature_importance: bool,
    incluir_top_regras: bool,
    incluir_serie_bruta: bool,
    label_encoder,
    datas_serie: list[str],
    colunas_serie: list[str],
    motivos_especialista: list[str],
) -> tuple[dict, dict | None, str]:
    if modo == "bruto":
        prefixo = "llm_bruto"
    elif modo == "estatisticas":
        prefixo = "llm_est"
    else:
        prefixo = "llm_global"
    try:
        resp = executar_teste_llm(
            id_serie=str(row_original["id_serie"]),
            valores_serie=valores_serie,
            modelo=modelo,
            tipo_algoritmo=tipo_algoritmo,
            nome_algoritmo=nome_algoritmo,
            classes=classes,
            df_matriz=df_matriz,
            idx_train=idx_train,
            feature_cols=feature_cols,
            incluir_feature_importance=incluir_feature_importance,
            incluir_top_regras=incluir_top_regras,
            label_encoder=label_encoder,
            datas_serie=datas_serie,
            modo_entrada_llm=modo,
            incluir_serie_bruta=incluir_serie_bruta,
        )
        prompt_capturado = resp.pop("_prompt", "")
        motivos_llm = resp.get("motivos_identificados_pelo_llm", [])
        pontos_llm = preparar_pontos_llm_para_visualizacao(
            pontos_llm=resp.get("llm_pontos", []),
            valores_serie=valores_serie,
            colunas_serie=colunas_serie,
        )
        comp = comparar_motivos(motivos_especialista, motivos_llm)
        campos = {
            f"{prefixo}_pontos":          _json_seguro(pontos_llm),
            f"{prefixo}_top_20":          _json_seguro(selecionar_top_pontos(pontos_llm, top_n=20)),
            f"{prefixo}_previsto":         resp.get("classe_interpretada_pelo_llm", ""),
            f"{prefixo}_confianca":        resp.get("nivel_confianca_llm", ""),
            f"{prefixo}_explicacao":       resp.get("explicacao_do_llm", ""),
            f"{prefixo}_analise":          resp.get("analise_global_da_serie", ""),
            f"{prefixo}_motivos":          _json_seguro(motivos_llm),
            f"{prefixo}_alinhamento":      comp["alinhamento_llm_especialista"],
            f"{prefixo}_score":            comp["score_alinhamento_motivos"],
            f"{prefixo}_motivos_em_comum": _json_seguro(comp["motivos_em_comum"]),
            f"{prefixo}_motivos_ausentes": _json_seguro(comp["motivos_ausentes_no_llm"]),
            f"{prefixo}_motivos_extras":   _json_seguro(comp["motivos_extras_do_llm"]),
            f"{prefixo}_erro":             "",
        }
        return campos, {**resp, "modo_entrada": modo}, prompt_capturado
    except Exception as exc:
        return {**_campos_llm_vazios(prefixo), f"{prefixo}_erro": str(exc)}, None, ""


def executar_pipeline(
    incluir_feature_importance: bool = True,
    incluir_top_regras: bool = True,
    incluir_serie_bruta: bool = False,
    algoritmos_llm: list[str] | None = None,
) -> tuple[pd.DataFrame, dict, dict, dict]:
    if algoritmos_llm is None:
        algoritmos_llm = ["random_forest"]
    load_dotenv(override=True)

    df = pd.read_csv(ARQUIVO_BASE) if ARQUIVO_BASE.exists() else gerar_base_sintetica()

    resultados_treino, df_matriz = treinar_todos_os_modelos(df)

    colunas_serie = obter_colunas_serie(df)
    datas_serie = [col.replace("dia_", "").replace("_", "-") for col in colunas_serie]

    if ARQUIVO_EVENTOS.exists():
        df_eventos = pd.read_csv(ARQUIVO_EVENTOS)
        mask = df_eventos["tem_evento"].astype(str).str.lower() == "true"

    resultados: list[dict] = []
    resultados_llm: list[dict] = []
    prompts_exemplo: dict[str, str] = {}

    exemplos_por_algoritmo: dict[str, dict] = {}
    for _tipo_alg, _pacote in resultados_treino.items():
        try:
            ex_a = selecionar_exemplo_classe(
                df_matriz, _pacote["idx_train"], _pacote["feature_cols"],
                _pacote["modelo"], "perfil_A", _tipo_alg,
                _pacote["classes"], _pacote.get("label_encoder"),
            )
            ex_b = selecionar_exemplo_classe(
                df_matriz, _pacote["idx_train"], _pacote["feature_cols"],
                _pacote["modelo"], "perfil_B", _tipo_alg,
                _pacote["classes"], _pacote.get("label_encoder"),
            )
            exemplos_por_algoritmo[_tipo_alg] = {"perfil_A": ex_a, "perfil_B": ex_b}
        except Exception:
            exemplos_por_algoritmo[_tipo_alg] = {}

    idx_test = next(iter(resultados_treino.values()))["idx_test"]

    for tipo_algoritmo, pacote in resultados_treino.items():
        modelo       = pacote["modelo"]
        feature_cols = pacote["feature_cols"]
        idx_train    = pacote["idx_train"]
        classes      = pacote["classes"]
        nome_algoritmo = pacote["nome_algoritmo"]
        label_encoder  = pacote.get("label_encoder")

        suporta_shap = tipo_algoritmo == "random_forest"
        suporta_lime = tipo_algoritmo == "random_forest"

        x_treino = df_matriz.loc[idx_train, feature_cols]

        for idx_original in idx_test:
            row_original = df.loc[idx_original]
            valores_serie = [int(row_original[col]) for col in colunas_serie]
            x_linha = df_matriz.loc[[idx_original], feature_cols]

            pred_classe  = _pred_classe(modelo, x_linha, tipo_algoritmo, label_encoder, classes)
            probabilidades = _pred_probas(modelo, x_linha, tipo_algoritmo, classes, label_encoder)

            motivos_especialista = json.loads(row_original.get("motivos_especialista", "[]"))

            pontos_shap: list[dict] = []
            pontos_lime: list[dict] = []
            erro_xai = ""

            try:
                if suporta_shap:
                    pontos_shap = calcular_pontos_shap(
                        modelo_rf=modelo,
                        x_linha=x_linha,
                        feature_cols=feature_cols,
                        classe_predita=pred_classe,
                    )
                if suporta_lime:
                    pontos_lime = calcular_pontos_lime(
                        modelo_rf=modelo,
                        x_treino=x_treino,
                        x_linha=x_linha,
                        feature_cols=feature_cols,
                        classe_predita=pred_classe,
                    )
            except Exception as exc:
                erro_xai = str(exc)

            resultado: dict = {
                "id_serie":               row_original["id_serie"],
                "classe_real":            row_original["classe_real"],
                "tipo_algoritmo":         tipo_algoritmo,
                "nome_algoritmo":         nome_algoritmo,
                "algoritmo_previsto":     pred_classe,
                "probabilidades_algoritmo": _json_seguro(probabilidades),
                "serie_temporal":         _json_seguro(valores_serie),
                "explicacao_ground_truth": row_original.get("explicacao_ground_truth", ""),
                "motivos_especialista":   _json_seguro(motivos_especialista),
                "feature_importance_incluida": incluir_feature_importance,
                "shap_pontos":  _json_seguro(pontos_shap),
                "lime_pontos":  _json_seguro(pontos_lime),
                "shap_top_20":  _json_seguro(selecionar_top_pontos(pontos_shap, top_n=20)),
                "lime_top_20":  _json_seguro(selecionar_top_pontos(pontos_lime, top_n=20)),
                "erro_xai":     erro_xai,
                **_campos_llm_vazios("llm_bruto"),
                **_campos_llm_vazios("llm_est"),
                **_campos_llm_vazios("llm_global"),
            }

            kwargs_llm = dict(
                row_original=row_original,
                valores_serie=valores_serie,
                modelo=modelo,
                tipo_algoritmo=tipo_algoritmo,
                nome_algoritmo=nome_algoritmo,
                classes=classes,
                df_matriz=df_matriz,
                idx_train=idx_train,
                feature_cols=feature_cols,
                incluir_feature_importance=incluir_feature_importance,
                incluir_top_regras=incluir_top_regras,
                incluir_serie_bruta=incluir_serie_bruta,
                label_encoder=label_encoder,
                datas_serie=datas_serie,
                colunas_serie=colunas_serie,
                motivos_especialista=motivos_especialista,
            )
            if tipo_algoritmo in algoritmos_llm:
                campos_bruto,  resp_bruto,  prompt_bruto  = _rodar_llm("bruto",        **kwargs_llm)
                campos_est,    resp_est,    prompt_est    = _rodar_llm("estatisticas", **kwargs_llm)
                campos_global, resp_global, prompt_global = _rodar_llm("global",       **kwargs_llm)

                resultado.update(campos_bruto)
                resultado.update(campos_est)
                resultado.update(campos_global)

                if resp_bruto:
                    resultados_llm.append(resp_bruto)
                if resp_est:
                    resultados_llm.append(resp_est)
                if resp_global:
                    resultados_llm.append(resp_global)

                if "bruto" not in prompts_exemplo and prompt_bruto:
                    prompts_exemplo["bruto"] = prompt_bruto
                if "estatisticas" not in prompts_exemplo and prompt_est:
                    prompts_exemplo["estatisticas"] = prompt_est
                if "global" not in prompts_exemplo and prompt_global:
                    prompts_exemplo["global"] = prompt_global

            resultados.append(resultado)

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(ARQUIVO_RESULTADO_COMPARACAO, index=False, encoding="utf-8-sig")
    pd.DataFrame(resultados_llm).to_csv(ARQUIVO_RELATORIO_LLM, index=False, encoding="utf-8-sig")

    return df_resultados, resultados_treino, prompts_exemplo, exemplos_por_algoritmo


if __name__ == "__main__":
    df_saida, _, _, _ = executar_pipeline()
    print(df_saida.head())
    print(f"Resultados salvos em: {ARQUIVO_RESULTADO_COMPARACAO}")
