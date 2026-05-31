from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import ARQUIVO_BASE, ARQUIVO_EVENTOS


def gerar_explicacao_comportamento(grupo: str, tem_outlier: bool) -> str:
    if grupo == "perfil_A":
        explicacao = (
            "Aluno com rotina de estudo constante e regular, sem grandes variações. "
            "O padrão indica esforço diário homogêneo e preparação distribuída ao longo do tempo."
        )
    elif grupo == "perfil_B":
        explicacao = (
            "Aluno com estudos concentrados próximo às avaliações, apresentando picos fortes em torno das provas. "
            "Comportamento típico de preparação focada em janelas curtas antes de atividades importantes."
        )
    else:
        explicacao = "Aluno com comportamento de estudo variável."

    if tem_outlier:
        explicacao += (
            " Alguns valores estão fora do limite esperado, sugerindo registros atípicos."
        )

    return explicacao


def gerar_motivos_especialista(grupo: str, tem_outlier: bool) -> list[str]:
    motivos: list[str] = []

    if grupo == "perfil_A":
        motivos.extend([
            "regularidade_constante",
            "ausencia_de_picos_pre_avaliacao",
        ])
    elif grupo == "perfil_B":
        motivos.extend([
            "picos_proximos_a_avaliacoes",
            "estudo_concentrado_em_janelas_curtas",
        ])

    if tem_outlier:
        motivos.append("presenca_de_outlier")

    return motivos


def gerar_base_sintetica(seed: int = 42, total_por_grupo: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    datas = pd.date_range("2026-02-01", "2026-07-31", freq="D")
    eventos = {
        "2026-03-10": "Trabalho 1",
        "2026-04-15": "Prova 1",
        "2026-06-05": "Prova 2",
        "2026-07-10": "Trabalho 2",
    }

    linhas: list[dict] = []

    for grupo in ["perfil_A", "perfil_B"]:
        alunos_com_outlier = set(rng.choice(range(1, total_por_grupo + 1), size=3, replace=False))

        for i in range(1, total_por_grupo + 1):
            id_serie = f"{grupo}_aluno_{i:02d}"

            if grupo == "perfil_A":
                valores = rng.normal(loc=10, scale=1.8, size=len(datas))
            else:
                valores = rng.normal(loc=5, scale=2.2, size=len(datas))
                for data_evento in eventos:
                    data_evento = pd.Timestamp(data_evento)
                    for deslocamento in range(-4, 2):
                        data_alvo = data_evento + pd.Timedelta(days=deslocamento)
                        if data_alvo in datas:
                            idx = datas.get_loc(data_alvo)
                            valores[idx] += rng.normal(loc=8, scale=3)

            valores = np.round(valores).astype(int)
            tem_outlier = i in alunos_com_outlier

            if tem_outlier:
                for _ in range(rng.integers(1, 4)):
                    idx = rng.integers(0, len(datas))
                    valores[idx] = int(rng.choice([-3, -2, -1, 16, 18, 20, 25]))

            linha = {
                "id_serie": id_serie,
                "classe_real": grupo,
                "explicacao_ground_truth": gerar_explicacao_comportamento(grupo, tem_outlier),
                "motivos_especialista": json.dumps(
                    gerar_motivos_especialista(grupo, tem_outlier),
                    ensure_ascii=False,
                ),
            }

            for data, valor in zip(datas, valores):
                linha[f"dia_{data.strftime('%Y_%m_%d')}"] = int(valor)

            linhas.append(linha)

    df = pd.DataFrame(linhas)
    df.to_csv(ARQUIVO_BASE, index=False, encoding="utf-8-sig")

    pd.DataFrame({
        "data": datas,
        "evento_avaliativo": [eventos.get(str(d.date()), "") for d in datas],
        "tem_evento": [str(d.date()) in eventos for d in datas],
    }).to_csv(ARQUIVO_EVENTOS, index=False, encoding="utf-8-sig")

    return df


if __name__ == "__main__":
    base = gerar_base_sintetica()
    print(f"Base criada com {len(base)} séries em {ARQUIVO_BASE}")
