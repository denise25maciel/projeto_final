from pathlib import Path

DATA_DIR = Path("data")
OUTPUTS_DIR = Path("outputs")
RESULTADOS_DIR = OUTPUTS_DIR / "resultados"

for pasta in [DATA_DIR, OUTPUTS_DIR, RESULTADOS_DIR]:
    pasta.mkdir(parents=True, exist_ok=True)

ARQUIVO_BASE = DATA_DIR / "series_temporais_educacionais.csv"
ARQUIVO_EVENTOS = DATA_DIR / "eventos_avaliativos.csv"
ARQUIVO_RESULTADO_COMPARACAO = RESULTADOS_DIR / "comparacao_resultados.csv"
ARQUIVO_MODELO_RF = RESULTADOS_DIR / "modelo_random_forest.joblib"

ARQUIVO_MATRIZ_RF = RESULTADOS_DIR / "matriz_entrada_random_forest.csv"

ARQUIVO_RELATORIO_LLM    = RESULTADOS_DIR / "resultado_llm_resumo.csv"
ARQUIVO_RELATORIO_LLM_RF = RESULTADOS_DIR / "resultado_llm_random_forest.csv"

ARQUIVOS_RELATORIO_LLM_POR_ALGORITMO = {
    "random_forest": ARQUIVO_RELATORIO_LLM_RF,
}

MODELO_LLM_PADRAO = "gpt-4.1-mini"
