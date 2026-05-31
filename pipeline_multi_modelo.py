from __future__ import annotations

"""
Pipeline principal multi-algoritmo.

Fluxo:
  1. Treina o Random Forest.
  2. Exibe as métricas do modelo.
  3. Pergunta ao usuário se quer incluir o feature importance no prompt enviado ao LLM.
  4. Executa a análise LLM para todas as séries de teste.
  5. Salva os resultados em arquivo CSV.
"""

import json
import sys

import pandas as pd

from config import ARQUIVO_BASE, ARQUIVO_EVENTOS, ARQUIVOS_RELATORIO_LLM_POR_ALGORITMO
from teste_llm_modelos import executar_teste_llm
from treinar_modelos import exibir_tabela_metricas, treinar_todos_os_modelos


# ---------------------------------------------------------------------------
# Interação com o usuário
# ---------------------------------------------------------------------------

def _perguntar_sim_nao(mensagem: str) -> bool:
    while True:
        resp = input(mensagem).strip().lower()
        if resp in ("s", "sim", "y", "yes"):
            return True
        if resp in ("n", "nao", "não", "no"):
            return False
        print("  Resposta inválida. Digite 's' para sim ou 'n' para não.")


def _coletar_preferencias_fi(resultados: dict) -> dict[str, bool]:
    print("\n" + "=" * 60)
    print("  CONFIGURAÇÃO DA ANÁLISE LLM")
    print("=" * 60)
    print(
        "Para cada algoritmo, indique se deseja enviar o seu\n"
        "feature importance ao LLM (s = sim, n = não).\n"
        "O feature importance enviado é a importância Gini/MDI do Random Forest.\n"
    )
    preferencias: dict[str, bool] = {}
    for chave, pacote in resultados.items():
        preferencias[chave] = _perguntar_sim_nao(
            f"  [{pacote['nome_algoritmo']}] Incluir feature importance? (s/n): "
        )
    return preferencias


# ---------------------------------------------------------------------------
# Execução por algoritmo
# ---------------------------------------------------------------------------

def _carregar_datas_serie(df_matriz: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    return [col.replace("dia_", "").replace("_", "-") for col in feature_cols]


def _executar_algoritmo(
    chave: str,
    pacote: dict,
    df_matriz: pd.DataFrame,
    incluir_fi: bool,
    modelo_llm: str,
) -> list[dict]:
    modelo = pacote["modelo"]
    feature_cols = pacote["feature_cols"]
    idx_train = pacote["idx_train"]
    idx_test = pacote["idx_test"]
    tipo_algoritmo = pacote["tipo_algoritmo"]
    nome_algoritmo = pacote["nome_algoritmo"]
    classes = pacote["classes"]
    label_encoder = pacote.get("label_encoder")

    datas_serie = _carregar_datas_serie(df_matriz, feature_cols)

    series_teste = df_matriz.loc[idx_test, ["id_serie"] + feature_cols]
    total = len(series_teste)
    resultados: list[dict] = []

    print(f"\n  Processando {total} séries de teste...")

    for i, (_, row) in enumerate(series_teste.iterrows(), start=1):
        id_serie = str(row["id_serie"])
        valores_serie = [int(row[col]) for col in feature_cols]

        try:
            resultado = executar_teste_llm(
                id_serie=id_serie,
                valores_serie=valores_serie,
                modelo=modelo,
                tipo_algoritmo=tipo_algoritmo,
                nome_algoritmo=nome_algoritmo,
                classes=classes,
                df_matriz=df_matriz,
                idx_train=idx_train,
                feature_cols=feature_cols,
                incluir_feature_importance=incluir_fi,
                label_encoder=label_encoder,
                datas_serie=datas_serie,
                modelo_llm=modelo_llm,
            )
            resultados.append(resultado)
            status = "✓"
        except Exception as exc:
            print(f"    [ERRO] {id_serie}: {exc}")
            resultados.append({"id_serie": id_serie, "erro": str(exc)})
            status = "✗"

        print(f"    [{i:>3}/{total}] {id_serie} {status}", end="\r", flush=True)

    print()
    return resultados


def _salvar_resultados(chave: str, resultados: list[dict]) -> None:
    if not resultados:
        return

    arquivo = ARQUIVOS_RELATORIO_LLM_POR_ALGORITMO[chave]

    linhas = []
    for r in resultados:
        linhas.append({
            "id_serie": r.get("id_serie", ""),
            "tipo_algoritmo": r.get("tipo_algoritmo", chave),
            "nome_algoritmo": r.get("nome_algoritmo", ""),
            "feature_importance_incluida": r.get("feature_importance_incluida", ""),
            "classe_predita_pelo_algoritmo": r.get("classe_predita_pelo_algoritmo", ""),
            "probabilidades_algoritmo": json.dumps(
                r.get("probabilidades_algoritmo", {}), ensure_ascii=False
            ),
            "classe_interpretada_pelo_llm": r.get("classe_interpretada_pelo_llm", ""),
            "nivel_confianca_llm": r.get("nivel_confianca_llm", ""),
            "motivos_identificados_pelo_llm": json.dumps(
                r.get("motivos_identificados_pelo_llm", []), ensure_ascii=False
            ),
            "explicacao_do_llm": r.get("explicacao_do_llm", ""),
            "llm_pontos": json.dumps(r.get("llm_pontos", []), ensure_ascii=False),
            "observacao_metodologica": r.get("observacao_metodologica", ""),
            "erro": r.get("erro", ""),
        })

    pd.DataFrame(linhas).to_csv(arquivo, index=False, encoding="utf-8-sig")
    print(f"  Resultados salvos em: {arquivo}")


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def main(modelo_llm: str = "gpt-4.1-mini") -> None:
    print("=" * 60)
    print("  PIPELINE MULTI-ALGORITMO — ESTUDO COMPARATIVO")
    print("=" * 60)

    # 1. Treinamento
    print("\nCarregando dados e treinando modelos...", flush=True)
    try:
        resultados, df_matriz = treinar_todos_os_modelos()
    except FileNotFoundError:
        print(
            "\n[ERRO] Arquivo de base não encontrado. "
            "Execute primeiro: python gerar_base.py"
        )
        sys.exit(1)

    # 2. Métricas
    exibir_tabela_metricas(resultados)

    # 3. Preferências de feature importance por algoritmo
    preferencias_fi = _coletar_preferencias_fi(resultados)

    # 4. Resumo das escolhas
    print("\n" + "-" * 60)
    print("  RESUMO DAS CONFIGURAÇÕES")
    print("-" * 60)
    for chave, pacote in resultados.items():
        fi_texto = "SIM" if preferencias_fi[chave] else "NÃO"
        print(f"  {pacote['nome_algoritmo']:<25}  Feature Importance: {fi_texto}")
    print("-" * 60)

    continuar = _perguntar_sim_nao("\nIniciar análise LLM com essas configurações? (s/n): ")
    if not continuar:
        print("Análise cancelada pelo usuário.")
        return

    # 5. Execução LLM por algoritmo
    for chave, pacote in resultados.items():
        nome = pacote["nome_algoritmo"]
        incluir_fi = preferencias_fi[chave]

        print(f"\n{'='*60}")
        print(f"  Algoritmo : {nome}")
        print(f"  Acurácia  : {pacote['accuracy']:.4f}  |  F1: {pacote['f1_weighted']:.4f}")
        print(f"  FI no LLM : {'Sim' if incluir_fi else 'Não'}")
        print("=" * 60)

        resultados_llm = _executar_algoritmo(
            chave=chave,
            pacote=pacote,
            df_matriz=df_matriz,
            incluir_fi=incluir_fi,
            modelo_llm=modelo_llm,
        )
        _salvar_resultados(chave, resultados_llm)

    print("\n" + "=" * 60)
    print("  ANÁLISE CONCLUÍDA")
    print("=" * 60)
    print("Arquivos gerados em outputs/resultados/:")
    for chave, arquivo in ARQUIVOS_RELATORIO_LLM_POR_ALGORITMO.items():
        nome = resultados[chave]["nome_algoritmo"]
        print(f"  {nome:<25}  {arquivo.name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline multi-algoritmo com análise LLM.")
    parser.add_argument(
        "--modelo-llm",
        default="gpt-4.1-mini",
        help="Modelo LLM a usar (padrão: gpt-4.1-mini)",
    )
    args = parser.parse_args()
    main(modelo_llm=args.modelo_llm)
