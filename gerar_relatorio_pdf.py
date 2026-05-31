from __future__ import annotations

import json
from datetime import date
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd


def _parse_json(valor: Any, padrao: Any) -> Any:
    if valor is None:
        return padrao
    if isinstance(valor, (list, dict)):
        return valor
    try:
        return json.loads(valor)
    except Exception:
        return padrao


def _pontos_para_linhas(pontos: list[dict], max_pontos: int = 20) -> list[str]:
    linhas = []
    for p in pontos[:max_pontos]:
        dia = p.get("dia", "?")
        valor = p.get("valor", "?")
        contrib_raw = (
            p.get("contribuicao") if p.get("contribuicao") is not None
            else p.get("importancia")
        )
        sentido = p.get("sentido", "")
        motivo = str(p.get("motivo_associado", ""))[:70]
        partes = [f"Dia {dia}  Valor {valor}"]
        if contrib_raw is not None:
            try:
                partes.append(f"Contrib {float(contrib_raw):+.4f}")
            except (TypeError, ValueError):
                pass
        if sentido:
            partes.append(sentido)
        if motivo:
            partes.append(motivo)
        linhas.append("  - " + "  |  ".join(partes))
    return linhas or ["  Sem pontos registrados."]


_CORES_METODOS = {
    "SHAP-RF":    "#f59e0b",
    "LIME-RF":    "#7c3aed",
    "LLM bruto":  "#ec4899",
    "LLM estat.": "#16a34a",
    "LLM global": "#0284c7",
}


def _gerar_grafico_serie(
    valores: list[int],
    pontos_por_metodo: dict[str, list[dict]] | None = None,
    titulo: str = "",
    figsize: tuple[float, float] = (9, 2.8),
) -> BytesIO | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    dias = list(range(1, len(valores) + 1))
    ax.plot(dias, valores, color="#374151", linewidth=1.5, zorder=1, label="serie")
    if valores:
        media = sum(valores) / len(valores)
        ax.axhline(media, color="#9ca3af", linewidth=1, linestyle="--", alpha=0.6, label="media")

    if pontos_por_metodo:
        for metodo, pontos in pontos_por_metodo.items():
            cor = _CORES_METODOS.get(metodo, "#888888")
            xs, ys = [], []
            for p in pontos:
                try:
                    d = int(p.get("dia", 0))
                    if 1 <= d <= len(valores):
                        xs.append(d)
                        ys.append(valores[d - 1])
                except (TypeError, ValueError):
                    pass
            if xs:
                ax.scatter(xs, ys, color=cor, s=55, zorder=3, alpha=0.88, label=metodo)

    ax.set_title(titulo, fontsize=9, pad=3)
    ax.set_xlabel("Dia", fontsize=7)
    ax.set_ylabel("Valor", fontsize=7)
    ax.tick_params(labelsize=6)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 2:
        ax.legend(handles=handles, labels=labels, fontsize=6,
                  loc="upper right", ncol=3, framealpha=0.7)

    fig.tight_layout(pad=0.4)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _gerar_grafico_exemplos(
    ex_a: dict,
    ex_b: dict,
    nome_algoritmo: str = "",
) -> BytesIO | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
    for ax, ex, classe, cor in zip(
        axes,
        [ex_a, ex_b],
        ["perfil_A", "perfil_B"],
        ["#2563eb", "#16a34a"],
    ):
        valores = (ex or {}).get("valores_serie", [])
        if not valores:
            ax.set_visible(False)
            continue
        dias = list(range(1, len(valores) + 1))
        ax.plot(dias, valores, color=cor, linewidth=1.8)
        media = sum(valores) / len(valores)
        ax.axhline(media, color="#9ca3af", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_title(f"Exemplo {classe}", fontsize=9, pad=3)
        ax.set_xlabel("Dia", fontsize=7)
        ax.set_ylabel("Valor", fontsize=7)
        ax.tick_params(labelsize=6)

    if nome_algoritmo:
        fig.suptitle(nome_algoritmo, fontsize=9, y=1.03)
    fig.tight_layout(pad=0.5)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _gerar_grafico_todos_metodos(
    valores: list[int],
    pontos_por_metodo: dict[str, list[dict]],
    titulo: str = "",
) -> BytesIO | None:
    """Figura única com 5 subplots empilhados, um por método, sem texto de pontos."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    ORDEM = ["SHAP-RF", "LIME-RF", "LLM bruto", "LLM estat.", "LLM global"]
    LABELS = {
        "SHAP-RF":    "SHAP - RF",
        "LIME-RF":    "LIME - RF",
        "LLM bruto":  "LLM Bruto",
        "LLM estat.": "LLM Estat.",
        "LLM global": "LLM Global",
    }

    fig, axes = plt.subplots(5, 1, figsize=(9, 10), sharex=True)
    dias = list(range(1, len(valores) + 1))
    media = sum(valores) / len(valores) if valores else 0

    for ax, metodo in zip(axes, ORDEM):
        cor = _CORES_METODOS.get(metodo, "#888888")
        ax.plot(dias, valores, color="#374151", linewidth=1.2, zorder=1)
        ax.axhline(media, color="#9ca3af", linewidth=0.8, linestyle="--", alpha=0.55)

        xs, ys = [], []
        for p in (pontos_por_metodo.get(metodo) or []):
            try:
                d = int(p.get("dia", 0))
                if 1 <= d <= len(valores):
                    xs.append(d)
                    ys.append(valores[d - 1])
            except (TypeError, ValueError):
                pass
        if xs:
            ax.scatter(xs, ys, color=cor, s=45, zorder=3, alpha=0.9, label=metodo)

        ax.set_ylabel(LABELS[metodo], fontsize=7.5, labelpad=3)
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Dia", fontsize=7)
    if titulo:
        fig.suptitle(titulo, fontsize=9, y=1.005)
    fig.tight_layout(pad=0.5, h_pad=0.6)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _heatmap_pdf(
    data: np.ndarray,
    ylabels: list[str],
    xlabels: list[str],
    titulo: str = "",
    cmap: str = "YlOrRd",
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = ".2f",
) -> BytesIO | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    nr, nc = len(ylabels), len(xlabels)
    fig, ax = plt.subplots(figsize=(max(nc * 2.0, 6), max(nr * 0.85, 2.2)))
    lo = vmin if vmin is not None else float(np.nanmin(data))
    hi = vmax if vmax is not None else float(np.nanmax(data))
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=lo, vmax=hi)
    ax.set_xticks(range(nc))
    ax.set_xticklabels(xlabels, fontsize=8, rotation=18, ha="right")
    ax.set_yticks(range(nr))
    ax.set_yticklabels(ylabels, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    span = max(hi - lo, 1e-9)
    for i in range(nr):
        for j in range(nc):
            v = data[i, j]
            if not np.isnan(v):
                cor_txt = "white" if (v - lo) / span > 0.58 else "black"
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        fontsize=8, color=cor_txt)
    if titulo:
        ax.set_title(titulo, fontsize=9, pad=5)
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _abs_contrib(p: dict) -> float:
    return abs(float(p.get("contribuicao") or p.get("importancia") or 0))


def _top_pct_pontos(pontos: list[dict], pct: float) -> list[dict]:
    """Retorna os pct% superiores dos pontos ordenados por |contribuicao|."""
    if not pontos:
        return pontos
    ordenados = sorted(pontos, key=_abs_contrib, reverse=True)
    n_keep = max(1, round(len(ordenados) * pct))
    return ordenados[:n_keep]


def _top75_pontos(pontos: list[dict]) -> list[dict]:
    return _top_pct_pontos(pontos, 0.75)


def _info_filtro(pontos_raw: list[dict], pontos_filtrados: list[dict], pct: float) -> str:
    """Texto resumido: quantos pontos exibidos, limiar de contribuicao."""
    n_tot = len(pontos_raw)
    n_fil = len(pontos_filtrados)
    if pontos_filtrados:
        limiar = min(_abs_contrib(p) for p in pontos_filtrados)
    else:
        limiar = 0.0
    return (
        f"{int(pct * 100)}% superiores: {n_fil}/{n_tot} pontos  "
        f"| limiar |contrib| >= {limiar:.4f}"
    )


def _resumo_contrib(pontos: list[dict]) -> tuple[float, float, float, float]:
    """Retorna (min_all, max_all, min_top75, max_top75) de |contribuicao|."""
    vals = [_abs_contrib(p) for p in pontos]
    if not vals:
        return (0.0, 0.0, 0.0, 0.0)
    top75 = sorted(vals, reverse=True)[:max(1, round(len(vals) * 0.75))]
    return (min(vals), max(vals), min(top75), max(top75))


def _gerar_grafico_sl_filtrado(
    valores: list[int],
    pts_shap: list[dict],
    pts_lime: list[dict],
    titulo: str = "",
) -> BytesIO | None:
    """Figura com 2 subplots (SHAP e LIME) com os pontos já filtrados recebidos."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    dias = list(range(1, len(valores) + 1))
    media = sum(valores) / len(valores) if valores else 0

    fig, (ax_s, ax_l) = plt.subplots(2, 1, figsize=(9, 5), sharex=True, sharey=True)

    for ax, pts, cor, label in [
        (ax_s, pts_shap, _CORES_METODOS["SHAP-RF"], "SHAP - RF"),
        (ax_l, pts_lime, _CORES_METODOS["LIME-RF"],  "LIME - RF"),
    ]:
        ax.plot(dias, valores, color="#374151", linewidth=1.2, zorder=1)
        ax.axhline(media, color="#9ca3af", linewidth=0.8, linestyle="--", alpha=0.55)
        xs, ys = [], []
        for p in pts:
            try:
                d = int(p.get("dia", 0))
                if 1 <= d <= len(valores):
                    xs.append(d)
                    ys.append(valores[d - 1])
            except (TypeError, ValueError):
                pass
        if xs:
            ax.scatter(xs, ys, color=cor, s=45, zorder=3, alpha=0.9)
        ax.set_ylabel(label, fontsize=7.5)
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_l.set_xlabel("Dia", fontsize=7)
    if titulo:
        fig.suptitle(titulo, fontsize=9, y=1.005)
    fig.tight_layout(pad=0.5, h_pad=0.5)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _calcular_metricas_xai(df_resultados: pd.DataFrame) -> dict:
    """Calcula as 4 metricas XAI. Retorna dict com DataFrames prontos."""
    METODOS = ["SHAP-RF", "LIME-RF", "LLM bruto", "LLM estat.", "LLM global"]
    MAPA_COL = {
        "SHAP-RF":    "shap_pontos",
        "LIME-RF":    "lime_pontos",
        "LLM bruto":  "llm_bruto_pontos",
        "LLM estat.": "llm_est_pontos",
        "LLM global": "llm_global_pontos",
    }
    QN = ["Q1 (1-45)", "Q2 (46-90)", "Q3 (91-135)", "Q4 (136-181)"]
    QR = [(1, 45), (46, 90), (91, 135), (136, 181)]
    EV = ["Picos extremos", "Vales / outliers", "Rupturas abruptas"]
    MC = ["SHAP-RF", "LIME-RF"]
    MD = ["LLM bruto", "LLM estat.", "LLM global"]

    df_u = df_resultados.drop_duplicates("id_serie")

    cont_q   = {m: {q: 0   for q in QN} for m in METODOS}
    tot_pts  = {m: 0 for m in METODOS}
    tot_ev   = {e: 0 for e in EV}
    det2     = {m: {e: 0   for e in EV} for m in METODOS}
    ipe_ss   = {m: {e: 0.0 for e in EV} for m in METODOS}
    ipe_sn   = {m: {e: 0   for e in EV} for m in METODOS}
    ipe_is   = {m: {e: 0.0 for e in EV} for m in MC}
    ipe_in   = {m: {e: 0   for e in EV} for m in MC}
    rk_soma  = {m: {e: 0.0 for e in EV} for m in MC}
    rk_n     = {m: {e: 0   for e in EV} for m in MC}
    det4     = {m: {e: 0   for e in EV} for m in MD}

    for _, row in df_u.iterrows():
        serie = _parse_json(row.get("serie_temporal", "[]"), [])
        if not serie or len(serie) < 2:
            continue
        vals  = np.array(serie, dtype=float)
        media = vals.mean()
        std   = vals.std()
        difs  = np.abs(np.diff(vals))
        std_d = difs.std()
        lim_r = 1.5 * std_d if std_d > 0 else float("inf")

        mev = {
            "Picos extremos":    {i + 1 for i, v in enumerate(vals) if v > media + 1.5 * std},
            "Vales / outliers":  {i + 1 for i, v in enumerate(vals) if v < media - 1.5 * std},
            "Rupturas abruptas": {i + 2 for i, d in enumerate(difs) if d > lim_r},
        }
        for e, de in mev.items():
            tot_ev[e] += len(de)

        for metodo in METODOS:
            col    = MAPA_COL.get(metodo, "")
            pontos = _parse_json(row.get(col, "[]"), [])
            dc: dict[int, float] = {}
            dr: dict[int, int]   = {}
            for p in pontos:
                d = p.get("dia")
                if d is None:
                    continue
                di = int(d)
                c  = p.get("contribuicao")
                r  = p.get("ranking")
                dc[di] = abs(float(c)) if c is not None else 0.0
                if r is not None:
                    try:
                        dr[di] = int(r)
                    except (TypeError, ValueError):
                        pass

            n_sel = len(dc)
            sel_set = set(dc)

            for di in sel_set:
                tot_pts[metodo] += 1
                for qn, (qi, qf) in zip(QN, QR):
                    if qi <= di <= qf:
                        cont_q[metodo][qn] += 1
                        break

            for e, de in mev.items():
                det2[metodo][e] += len(de & sel_set)
                n_ev = len(de)
                if n_ev > 0 and n_sel > 0:
                    fb  = n_ev / len(vals)
                    nse = len(de & sel_set)
                    ipe_ss[metodo][e] += (nse / n_sel) / fb
                    ipe_sn[metodo][e] += 1
                    if metodo in MC:
                        ce  = [dc[di] for di in sel_set if di in de]
                        cne = [dc[di] for di in sel_set if di not in de]
                        if ce and cne:
                            ipe_is[metodo][e] += float(np.mean(ce)) / float(np.mean(cne))
                            ipe_in[metodo][e] += 1

            if metodo in MC:
                for e, de in mev.items():
                    for di in de:
                        if di in dr:
                            rk_soma[metodo][e] += dr[di]
                            rk_n[metodo][e]    += 1
            if metodo in MD:
                for e, de in mev.items():
                    det4[metodo][e] += len(de & sel_set)

    # DataFrames
    df_m1 = pd.DataFrame(
        {m: {q: (cont_q[m][q] / tot_pts[m]) if tot_pts[m] > 0 else 0.0 for q in QN}
         for m in METODOS}, index=QN,
    ).T

    m2r = [{"Evento": e, "Total": tot_ev[e],
             **{m: det2[m][e] / tot_ev[e] if tot_ev[e] > 0 else np.nan for m in METODOS}}
           for e in EV]
    df_m2 = pd.DataFrame(m2r).set_index("Evento")

    df_m3s = pd.DataFrame(
        {m: {e: ipe_ss[m][e] / ipe_sn[m][e] if ipe_sn[m][e] > 0 else np.nan for e in EV}
         for m in METODOS}, index=EV,
    ).T

    m3i = [{"Evento": e,
             **{m: ipe_is[m][e] / ipe_in[m][e] if ipe_in[m][e] > 0 else np.nan for m in MC}}
           for e in EV]
    df_m3i = pd.DataFrame(m3i).set_index("Evento")

    m4r = [{"Evento": e,
             **{m: rk_soma[m][e] / rk_n[m][e] if rk_n[m][e] > 0 else np.nan for m in MC}}
           for e in EV]
    df_m4r = pd.DataFrame(m4r).set_index("Evento")

    m4d = [{"Evento": e,
             **{m: det4[m][e] / tot_ev[e] if tot_ev[e] > 0 else np.nan for m in MD}}
           for e in EV]
    df_m4d = pd.DataFrame(m4d).set_index("Evento")

    return dict(
        df_m1=df_m1, df_m2=df_m2,
        df_m3s=df_m3s, df_m3i=df_m3i,
        df_m4r=df_m4r, df_m4d=df_m4d,
        METODOS=METODOS, QN=QN, EV=EV, MC=MC, MD=MD,
        n_series=len(df_u),
    )


_SUBSTITUICOES_LATIN1 = str.maketrans({
    "—": "--",   # em dash
    "–": "-",    # en dash
    "‘": "'",    # aspas simples esquerda
    "’": "'",    # aspas simples direita
    "“": '"',    # aspas duplas esquerda
    "”": '"',    # aspas duplas direita
    "…": "...",  # reticencias
    "·": "*",    # middle dot
    "•": "-",    # bullet
})


def _latin1(texto: str) -> str:
    return texto.translate(_SUBSTITUICOES_LATIN1).encode("latin-1", errors="replace").decode("latin-1")


def gerar_relatorio_pdf(
    df_resultados: pd.DataFrame,
    parametros: dict,
    prompts_exemplo: dict[str, str] | None = None,
    exemplos_por_algoritmo: dict[str, dict] | None = None,
) -> bytes:
    try:
        from fpdf import FPDF
    except ImportError as exc:
        raise ImportError("Instale fpdf2:  pip install fpdf2") from exc

    MARGEM = 15
    LARGURA = 210 - 2 * MARGEM

    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "I", 7)
            self.cell(
                0, 5,
                "Relatorio de Explicabilidade XAI - Series Temporais Educacionais",
                align="C", new_x="LMARGIN", new_y="NEXT",
            )
            self.set_draw_color(200, 200, 200)
            self.line(MARGEM, self.get_y(), 210 - MARGEM, self.get_y())
            self.ln(2)

        def footer(self):
            self.set_y(-13)
            self.set_font("Helvetica", "I", 7)
            self.cell(0, 5, f"Pagina {self.page_no()}", align="C")

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(left=MARGEM, top=14, right=MARGEM)

    def h1(texto: str) -> None:
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_fill_color(215, 215, 215)
        pdf.multi_cell(LARGURA, 7, _latin1(texto), fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    def h2(texto: str) -> None:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(238, 238, 238)
        pdf.multi_cell(LARGURA, 6, _latin1(texto), fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    def h3(texto: str) -> None:
        pdf.set_font("Helvetica", "B", 9)
        pdf.multi_cell(LARGURA, 5, _latin1(texto), new_x="LMARGIN", new_y="NEXT")

    def p(texto: str, tam: int = 9) -> None:
        pdf.set_font("Helvetica", "", tam)
        pdf.multi_cell(LARGURA, 5, _latin1(texto), new_x="LMARGIN", new_y="NEXT")

    def separador() -> None:
        pdf.set_draw_color(180, 180, 180)
        pdf.line(MARGEM, pdf.get_y(), 210 - MARGEM, pdf.get_y())
        pdf.ln(4)

    def inserir_imagem(buf: BytesIO | None, w: float | None = None) -> None:
        if buf is None:
            return
        buf.seek(0)
        pdf.image(buf, w=w or LARGURA)
        pdf.ln(3)

    sim_nao = lambda v: "Sim" if v else "Nao"

    # ─── Capa ─────────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(LARGURA, 10, "Relatorio de Explicabilidade XAI",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(LARGURA, 8, "Series Temporais Educacionais",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(LARGURA, 6, f"Gerado em: {date.today().isoformat()}",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    # ─── Parâmetros ───────────────────────────────────────────────────────────
    h1("PARAMETROS DO EXPERIMENTO")
    p(f"Feature importance enviada ao LLM:              {sim_nao(parametros.get('incluir_feature_importance', True))}")
    p(f"Caminhos de decisao enviados ao LLM:            {sim_nao(parametros.get('incluir_top_regras', True))}")
    p(f"Serie bruta incluida nos LLMs estatisticos:     {sim_nao(parametros.get('incluir_serie_bruta', False))}")
    algs_llm = parametros.get("algoritmos_llm", ["random_forest"])
    p(f"Algoritmos com LLM ativo:                       {', '.join(algs_llm) if algs_llm else 'nenhum'}")
    pdf.ln(5)

    # ─── Exemplos representativos de treino ───────────────────────────────────
    if exemplos_por_algoritmo:
        h1("EXEMPLOS REPRESENTATIVOS DE TREINO")
        _NOMES_ALG = {
            "random_forest": "Random Forest",
        }
        for tipo_alg, exemplos in exemplos_por_algoritmo.items():
            if not exemplos:
                continue
            nome_alg = _NOMES_ALG.get(tipo_alg, tipo_alg)
            h2(nome_alg)
            ex_a = exemplos.get("perfil_A", {})
            ex_b = exemplos.get("perfil_B", {})
            buf_ex = _gerar_grafico_exemplos(ex_a, ex_b, nome_alg)
            inserir_imagem(buf_ex)
        pdf.ln(3)

    # ─── Prompt exemplo ───────────────────────────────────────────────────────
    if prompts_exemplo:
        h1("EXEMPLO DE PROMPT ENVIADO AO LLM")
        for modo_label, prompt_texto in prompts_exemplo.items():
            if not prompt_texto:
                continue
            h2(f"Modo: {modo_label}")
            exibir = prompt_texto[:5000] + ("\n[... truncado ...]" if len(prompt_texto) > 5000 else "")
            pdf.set_font("Courier", "", 7)
            pdf.multi_cell(LARGURA, 3.8, _latin1(exibir), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)

    # ─── Métricas XAI ────────────────────────────────────────────────────────
    try:
        mx = _calcular_metricas_xai(df_resultados)
    except Exception:
        mx = None

    if mx:
        METODOS = mx["METODOS"]
        QN      = mx["QN"]
        EV      = mx["EV"]
        MC      = mx["MC"]
        MD      = mx["MD"]

        def _tabela_simples(pdf_obj, df: pd.DataFrame, col_fmt: str,
                            largura: float, tam_fonte: int = 7) -> None:
            cols = list(df.columns)
            idx  = list(df.index)
            cw   = (largura - 38) / max(len(cols), 1)
            pdf_obj.set_font("Helvetica", "B", tam_fonte)
            pdf_obj.set_fill_color(210, 210, 210)
            pdf_obj.cell(38, 5, "Evento / Metodo", border=1, fill=True)
            for c in cols:
                pdf_obj.cell(cw, 5, _latin1(str(c)[:16]), border=1, fill=True)
            pdf_obj.ln()
            fill = False
            for i_name in idx:
                pdf_obj.set_fill_color(245, 245, 245) if fill else pdf_obj.set_fill_color(255, 255, 255)
                pdf_obj.set_font("Helvetica", "B" if fill else "", tam_fonte)
                pdf_obj.cell(38, 4.5, _latin1(str(i_name)[:22]), border=1, fill=fill)
                pdf_obj.set_font("Helvetica", "", tam_fonte)
                for c in cols:
                    v = df.loc[i_name, c]
                    if isinstance(v, float) and not np.isnan(v):
                        txt = format(v, col_fmt)
                    else:
                        txt = str(v) if not (isinstance(v, float) and np.isnan(v)) else "--"
                    pdf_obj.cell(cw, 4.5, _latin1(txt[:14]), border=1, fill=fill)
                pdf_obj.ln()
                fill = not fill
            pdf_obj.ln(2)

        pdf.add_page()
        h1("METRICAS XAI — COMPARACAO ENTRE METODOS DE EXPLICABILIDADE")
        p(
            f"As metricas abaixo foram calculadas sobre as {mx['n_series']} series "
            "do conjunto de teste. Cada metrica revela um angulo diferente de como "
            "SHAP, LIME e os tres modos de LLM selecionam pontos relevantes nas series.",
            tam=8,
        )
        pdf.ln(3)

        # ── Métrica 1 ──────────────────────────────────────────────────────
        h2("METRICA 1 — COBERTURA DE QUARTIS")
        p(
            "O QUE MEDE: Para cada metodo, que percentual de seus pontos selecionados "
            "cai em cada um dos 4 quartis temporais da serie "
            "(Q1=dias 1-45, Q2=46-90, Q3=91-135, Q4=136-181).",
            tam=8,
        )
        p(
            "COMO CALCULAR: Conta-se quantos dias selecionados pelo metodo pertencem "
            "a cada quartil e divide pelo total de dias selecionados por aquele metodo.",
            tam=8,
        )
        p(
            "EXEMPLO DIDATICO: O SHAP selecionou 20 pontos para o aluno Joao. "
            "Desses, 6 estavam no Q1, 4 no Q2, 5 no Q3 e 5 no Q4. "
            "Cobertura: Q1=30%, Q2=20%, Q3=25%, Q4=25%. "
            "A referencia de distribuicao uniforme e 25% em cada quartil. "
            "O desvio do Q1 (30% vs 25%) indica leve preferencia pelo inicio da serie.",
            tam=8,
        )
        p(
            "COMO INTERPRETAR: Verde/escuro = alta concentracao naquele quartil. "
            "Um metodo sem vies temporal mostraria ~25% em todos os quartis. "
            "Concentracoes revelam que o metodo 'prefere' certos periodos da serie.",
            tam=8,
        )
        pdf.ln(2)
        buf_m1 = _heatmap_pdf(
            mx["df_m1"].values.astype(float),
            METODOS, QN,
            titulo="Cobertura de quartis (proporcao dos pontos selecionados)",
            cmap="YlOrRd", vmin=0, vmax=1, fmt=".0%",
        )
        inserir_imagem(buf_m1)
        separador()

        # ── Métrica 2 ──────────────────────────────────────────────────────
        pdf.add_page()
        h2("METRICA 2 — TAXA DE DETECCAO DE EVENTOS ESTRUTURAIS")
        p(
            "O QUE MEDE: De todos os dias em que ocorreu um evento estrutural na serie "
            "(pico, vale ou ruptura), qual fracao foi selecionada por cada metodo.",
            tam=8,
        )
        p(
            "DEFINICAO DOS EVENTOS (limiar por serie):\n"
            "  Pico extremo:    valor > media + 1,5 x desvio-padrao\n"
            "  Vale / outlier:  valor < media - 1,5 x desvio-padrao\n"
            "  Ruptura abrupta: |valor[t] - valor[t-1]| > 1,5 x dp das diferencas",
            tam=8,
        )
        p(
            "COMO CALCULAR: Para cada serie, identificam-se os dias de evento. "
            "Verifica-se se o dia aparece entre os pontos selecionados pelo metodo. "
            "Soma-se as deteccoes e divide pelo total de dias de evento.",
            tam=8,
        )
        p(
            "EXEMPLO DIDATICO: A serie de Pedro tem media=8, dp=3. "
            "Limiar de pico = 8 + 1,5x3 = 12,5. Os dias 22, 45, 87 e 133 sao picos. "
            "O SHAP selecionou os dias 22 e 45 — taxa de 2/4 = 50%. "
            "O LLM Estat. selecionou os dias 22, 45, 87 e 133 — taxa de 4/4 = 100%. "
            "ATENCAO: esta metrica e binaria. Nao diz QUANTA importancia foi atribuida "
            "ao pico, apenas se ele foi incluido na selecao.",
            tam=8,
        )
        pdf.ln(2)
        df_m2_pct = mx["df_m2"].drop(columns=["Total"], errors="ignore")
        buf_m2 = _heatmap_pdf(
            df_m2_pct.values.astype(float),
            list(df_m2_pct.index), list(df_m2_pct.columns),
            titulo="Taxa de deteccao de eventos (fracao dos dias de evento selecionados)",
            cmap="Blues", vmin=0, vmax=1, fmt=".0%",
        )
        inserir_imagem(buf_m2)
        h3("Contagem total de eventos nas series de teste")
        tot_col = mx["df_m2"]["Total"] if "Total" in mx["df_m2"].columns else None
        if tot_col is not None:
            for ev, tot in tot_col.items():
                p(f"  {ev}: {int(tot)} dias de evento no conjunto de teste", tam=8)
        separador()

        # ── Métrica 3 ──────────────────────────────────────────────────────
        pdf.add_page()
        h2("METRICA 3 — INDICE DE PRIORIZACAO DE EVENTOS (IPE)")
        p(
            "O QUE MEDE: O quanto cada metodo concentra atencao nos eventos em "
            "relacao ao que seria esperado pelo acaso puro.",
            tam=8,
        )
        p(
            "IPE DE SELECAO (todos os metodos):\n"
            "  IPE = (fracao dos pontos selecionados que sao eventos)\n"
            "        / (fracao dos 181 dias que sao eventos)\n"
            "  IPE > 1,0 -> prioriza eventos\n"
            "  IPE = 1,0 -> mesmo que o acaso\n"
            "  IPE < 1,0 -> subestima eventos",
            tam=8,
        )
        p(
            "EXEMPLO IPE DE SELECAO: 10% dos 181 dias sao picos (18 dias). "
            "O SHAP selecionou 20 pontos, dos quais 3 sao picos (15%). "
            "IPE = 15% / 10% = 1,5 (prioriza picos). "
            "Se selecionou apenas 1 pico (5%), IPE = 0,5 (subestima picos).",
            tam=8,
        )
        p(
            "IPE DE IMPORTANCIA (somente SHAP e LIME):\n"
            "  IPE_imp = media(|contribuicao| nos dias de evento selecionados)\n"
            "            / media(|contribuicao| nos dias sem evento selecionados)\n"
            "  IPE_imp < 1,0 indica que o metodo atribui MENOS importancia aos "
            "eventos do que aos dias comuns — mesmo quando os seleciona.",
            tam=8,
        )
        p(
            "EXEMPLO IPE DE IMPORTANCIA: O SHAP seleciona o dia 45 (pico) com "
            "contribuicao +0,018 e o dia 60 (normal) com contribuicao +0,062. "
            "Media nos picos = 0,020; media nos normais = 0,055. "
            "IPE_imp = 0,020 / 0,055 = 0,36. O modelo considera os dias normais "
            "quase 3x mais relevantes para a classificacao do que os picos.",
            tam=8,
        )
        pdf.ln(2)
        h3("IPE de Selecao — todos os metodos (referencia: 1,0 = acaso)")
        buf_m3s = _heatmap_pdf(
            mx["df_m3s"].values.astype(float),
            METODOS, EV,
            titulo="IPE de selecao (>1=prioriza, <1=subestima)",
            cmap="RdYlGn", vmin=0, vmax=max(2.0, float(np.nanmax(mx["df_m3s"].values))),
            fmt=".2f",
        )
        inserir_imagem(buf_m3s)
        h3("IPE de Importancia — SHAP-RF e LIME-RF (referencia: 1,0 = mesmo peso)")
        _tabela_simples(pdf, mx["df_m3i"], ".3f", LARGURA)
        separador()

        # ── Métrica 4 ──────────────────────────────────────────────────────
        pdf.add_page()
        h2("METRICA 4 — POSICAO DOS EVENTOS NO RANKING SHAP/LIME + TAXA DE DETECCAO LLMs")
        p(
            "MOTIVACAO: Ao visualizar os graficos das series, observa-se que SHAP e LIME "
            "tendem a selecionar pontos proximos a media, enquanto alguns LLMs identificam "
            "diretamente os picos extremos. Esta metrica quantifica essa observacao visual.",
            tam=8,
        )
        p(
            "SHAP E LIME — POSICAO MEDIA NO RANKING:\n"
            "  O SHAP e o LIME atribuem uma contribuicao a todos os 181 dias e os "
            "ordenam do maior ao menor impacto absoluto (ranking 1 a 181). "
            "Para cada dia de evento, registra-se sua posicao nesse ranking e "
            "calcula-se a media entre todas as series.\n"
            "  Referencia: posicao 91 = acaso puro (esperado sem preferencia).\n"
            "  Posicao > 91 = eventos tratados como MENOS importantes que a media.\n"
            "  Posicao < 91 = eventos tratados como MAIS importantes que a media.",
            tam=8,
        )
        p(
            "EXEMPLO POSICAO NO RANKING: O pico no dia 45 (valor=15) aparece na "
            "posicao 120 do ranking SHAP — significa que 119 outros dias receberam "
            "mais atencao do modelo. A referencia e 91. Posicao 120 indica que o "
            "modelo Random Forest nao considera esse pico como evidencia discriminativa "
            "para classificar o aluno.",
            tam=8,
        )
        p(
            "LLMs — TAXA DE DETECCAO:\n"
            "  Os LLMs fazem uma selecao direta de 3 a 20 pontos. "
            "Calculamos: dos dias de evento, quantos foram incluidos na selecao?\n"
            "  0% = o LLM nunca seleciona dias de evento.\n"
            "  100% = o LLM sempre inclui todos os dias de evento.",
            tam=8,
        )
        p(
            "LEITURA CONJUNTA DAS DUAS TABELAS: Se o SHAP posiciona picos na posicao "
            "media 115/181 (abaixo do acaso em 91) e o LLM Estat. detecta 70% dos "
            "mesmos picos, o contraste quantifica o argumento central do estudo: "
            "os metodos baseados em LLM capturam anomalias que o modelo de ML considera "
            "irrelevantes para a decisao de classificacao.",
            tam=8,
        )
        pdf.ln(2)
        h3("SHAP-RF e LIME-RF — posicao media dos eventos no ranking (referencia: 91 = acaso)")
        buf_m4r = _heatmap_pdf(
            mx["df_m4r"].values.astype(float),
            EV, MC,
            titulo="Posicao media no ranking (1=mais importante, 181=menos importante)",
            cmap="RdYlGn_r", vmin=1, vmax=181, fmt=".1f",
        )
        inserir_imagem(buf_m4r)
        h3("LLMs — taxa de deteccao dos eventos estruturais")
        buf_m4d = _heatmap_pdf(
            mx["df_m4d"].values.astype(float),
            EV, MD,
            titulo="Taxa de deteccao pelos LLMs (fracao dos dias de evento selecionados)",
            cmap="Blues", vmin=0, vmax=1, fmt=".0%",
        )
        inserir_imagem(buf_m4d)
        separador()

    # ─── Pontuação SHAP e LIME por série ─────────────────────────────────────
    df_rf_sl = df_resultados[df_resultados["tipo_algoritmo"] == "random_forest"].copy()
    if not df_rf_sl.empty:
        pdf.add_page()
        h1("PONTUACAO SHAP E LIME — MINIMO, MAXIMO E TOP-75%")
        p(
            "Para cada serie do conjunto de teste sao exibidos o valor minimo e "
            "maximo de |contribuicao| atribuido pelo SHAP e pelo LIME a todos os "
            "181 dias, alem dos mesmos limites considerando apenas os 75% dos pontos "
            "com pontuacao mais alta (top-75%).",
            tam=8,
        )
        p(
            "COMO INTERPRETAR O TOP-75%: Se o SHAP atribuiu scores a 181 dias e "
            "os ordenou do maior ao menor, o top-75% corresponde aos 136 dias de "
            "maior contribuicao absoluta. O minimo do top-75% e o limiar abaixo do "
            "qual os 25% menos relevantes foram descartados.",
            tam=8,
        )
        pdf.ln(2)

        _COLS_SL = [
            ("ID Serie",      20),
            ("Classe",        18),
            ("SHAP min",      22),
            ("SHAP max",      22),
            ("SHAP t75 min",  22),
            ("SHAP t75 max",  22),
            ("LIME min",      22),
            ("LIME max",      22),
            ("LIME t75 min",  18),
            ("LIME t75 max",  18),
        ]
        pdf.set_font("Helvetica", "B", 6)
        pdf.set_fill_color(210, 210, 210)
        for cab, w in _COLS_SL:
            pdf.cell(w, 5, cab, border=1, fill=True)
        pdf.ln()

        fill_sl = False
        for _, row_sl in df_rf_sl.iterrows():
            if pdf.get_y() > 262:
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 6)
                pdf.set_fill_color(210, 210, 210)
                for cab, w in _COLS_SL:
                    pdf.cell(w, 5, cab, border=1, fill=True)
                pdf.ln()

            pts_shap = _parse_json(row_sl.get("shap_pontos", "[]"), [])
            pts_lime = _parse_json(row_sl.get("lime_pontos",  "[]"), [])
            s_mn, s_mx, s_t75mn, s_t75mx = _resumo_contrib(pts_shap)
            l_mn, l_mx, l_t75mn, l_t75mx = _resumo_contrib(pts_lime)

            pdf.set_fill_color(245, 245, 245) if fill_sl else pdf.set_fill_color(255, 255, 255)
            pdf.set_font("Helvetica", "", 6)
            vals_row = [
                str(row_sl.get("id_serie", ""))[:18],
                str(row_sl.get("classe_real", ""))[:14],
                f"{s_mn:.4f}", f"{s_mx:.4f}", f"{s_t75mn:.4f}", f"{s_t75mx:.4f}",
                f"{l_mn:.4f}", f"{l_mx:.4f}", f"{l_t75mn:.4f}", f"{l_t75mx:.4f}",
            ]
            for val_txt, (_, w) in zip(vals_row, _COLS_SL):
                pdf.cell(w, 4.5, _latin1(val_txt), border=1, fill=fill_sl)
            pdf.ln()
            fill_sl = not fill_sl

        pdf.ln(4)
        separador()

    # ─── Tabela comparativa de resultados ────────────────────────────────────
    df_rf = df_resultados[df_resultados["tipo_algoritmo"] == "random_forest"].copy()
    if not df_rf.empty:
        pdf.add_page()
        h1("RESULTADOS - TABELA COMPARATIVA")
        p("Previsoes e alinhamento de cada metodo para todas as series de teste.", tam=8)
        pdf.ln(2)

        _COLUNAS_TAB = [
            ("ID Serie",    "id_serie",                22),
            ("Cl. Real",    "classe_real",             20),
            ("RF Prev.",    "algoritmo_previsto",      20),
            ("LLM B.",      "llm_bruto_previsto",      18),
            ("LLM E.",      "llm_est_previsto",        18),
            ("LLM G.",      "llm_global_previsto",     18),
            ("Al. Bruto",   "llm_bruto_alinhamento",   22),
            ("Al. Estat.",  "llm_est_alinhamento",     22),
            ("Al. Global",  "llm_global_alinhamento",  20),
        ]

        pdf.set_font("Helvetica", "B", 7)
        pdf.set_fill_color(200, 200, 200)
        for cabecalho, _, w in _COLUNAS_TAB:
            pdf.cell(w, 5, cabecalho, border=1, fill=True)
        pdf.ln()

        fill = False
        for _, row in df_rf.iterrows():
            if pdf.get_y() > 260:
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 7)
                pdf.set_fill_color(200, 200, 200)
                for cabecalho, _, w in _COLUNAS_TAB:
                    pdf.cell(w, 5, cabecalho, border=1, fill=True)
                pdf.ln()
            pdf.set_fill_color(245, 245, 245) if fill else pdf.set_fill_color(255, 255, 255)
            pdf.set_font("Helvetica", "", 7)
            for _, col_df, w in _COLUNAS_TAB:
                val = _latin1(str(row.get(col_df, "") or "")[:22])
                pdf.cell(w, 4.5, val, border=1, fill=fill)
            pdf.ln()
            fill = not fill

        pdf.ln(4)
        separador()

    # ─── Resultados por série — apenas gráficos ───────────────────────────────
    ids_unicos = df_resultados["id_serie"].astype(str).unique().tolist()

    def _ordem(id_s: str) -> tuple:
        linhas = df_resultados[df_resultados["id_serie"].astype(str) == id_s]
        classe = str(linhas.iloc[0].get("classe_real", ""))
        return (0 if classe == "perfil_A" else 1, id_s)

    ids_ordenados = sorted(ids_unicos, key=_ordem)

    for id_serie in ids_ordenados:
        df_s = df_resultados[df_resultados["id_serie"].astype(str) == id_serie]

        mask_rf = df_s["tipo_algoritmo"] == "random_forest"
        linha_rf = df_s[mask_rf].iloc[0] if mask_rf.any() else None
        if linha_rf is None:
            continue

        valores    = _parse_json(linha_rf.get("serie_temporal", "[]"), [])
        classe_real = str(linha_rf.get("classe_real", "-"))

        pdf.add_page()
        h1(f"Serie: {id_serie}   |   Classe real: {classe_real}")

        prev_rf = str(linha_rf.get("algoritmo_previsto", "N/D"))
        llm_b   = str(linha_rf.get("llm_bruto_previsto",  "") or "N/D")
        llm_e   = str(linha_rf.get("llm_est_previsto",    "") or "N/D")
        llm_g   = str(linha_rf.get("llm_global_previsto", "") or "N/D")
        p(f"RF: {prev_rf}   LLM Bruto: {llm_b}   LLM Estat.: {llm_e}   LLM Global: {llm_g}")
        pdf.ln(2)

        pts_shap_raw = _parse_json(linha_rf.get("shap_pontos", "[]"), [])
        pts_lime_raw = _parse_json(linha_rf.get("lime_pontos", "[]"), [])
        pts_shap_t75 = _top75_pontos(pts_shap_raw)
        pts_lime_t75 = _top75_pontos(pts_lime_raw)

        s_mn, s_mx, s_t75mn, s_t75mx = _resumo_contrib(pts_shap_raw)
        l_mn, l_mx, l_t75mn, l_t75mx = _resumo_contrib(pts_lime_raw)
        p(
            f"SHAP: min={s_mn:.4f}  max={s_mx:.4f}  "
            f"top-75% [{s_t75mn:.4f} – {s_t75mx:.4f}]  "
            f"({len(pts_shap_t75)}/{len(pts_shap_raw)} pontos exibidos)   "
            f"LIME: min={l_mn:.4f}  max={l_mx:.4f}  "
            f"top-75% [{l_t75mn:.4f} – {l_t75mx:.4f}]  "
            f"({len(pts_lime_t75)}/{len(pts_lime_raw)} pontos exibidos)",
            tam=7,
        )
        pdf.ln(1)

        # ── Gráfico completo (todos os 5 métodos, SHAP/LIME top-75%) ──────────
        pontos_por_metodo = {
            "SHAP-RF":    pts_shap_t75,
            "LIME-RF":    pts_lime_t75,
            "LLM bruto":  _parse_json(linha_rf.get("llm_bruto_pontos",  "[]"), []),
            "LLM estat.": _parse_json(linha_rf.get("llm_est_pontos",    "[]"), []),
            "LLM global": _parse_json(linha_rf.get("llm_global_pontos", "[]"), []),
        }
        buf_comb = _gerar_grafico_todos_metodos(
            valores,
            pontos_por_metodo,
            titulo=f"{id_serie} - {classe_real}  |  SHAP/LIME: top-75%",
        )
        inserir_imagem(buf_comb)

        # ── Cópia SHAP + LIME — top-70% ───────────────────────────────────────
        pdf.add_page()
        h2(f"SHAP e LIME — 70% mais significativos  |  {id_serie} ({classe_real})")
        pts_shap_70 = _top_pct_pontos(pts_shap_raw, 0.70)
        pts_lime_70 = _top_pct_pontos(pts_lime_raw, 0.70)
        p(f"SHAP — {_info_filtro(pts_shap_raw, pts_shap_70, 0.70)}", tam=8)
        p(f"LIME — {_info_filtro(pts_lime_raw, pts_lime_70, 0.70)}", tam=8)
        pdf.ln(1)
        buf_70 = _gerar_grafico_sl_filtrado(
            valores, pts_shap_70, pts_lime_70,
            titulo=f"{id_serie} — top-70%",
        )
        inserir_imagem(buf_70)

        # ── Cópia SHAP + LIME — top-50% ───────────────────────────────────────
        h2(f"SHAP e LIME — 50% mais significativos  |  {id_serie} ({classe_real})")
        pts_shap_50 = _top_pct_pontos(pts_shap_raw, 0.50)
        pts_lime_50 = _top_pct_pontos(pts_lime_raw, 0.50)
        p(f"SHAP — {_info_filtro(pts_shap_raw, pts_shap_50, 0.50)}", tam=8)
        p(f"LIME — {_info_filtro(pts_lime_raw, pts_lime_50, 0.50)}", tam=8)
        pdf.ln(1)
        buf_50 = _gerar_grafico_sl_filtrado(
            valores, pts_shap_50, pts_lime_50,
            titulo=f"{id_serie} — top-50%",
        )
        inserir_imagem(buf_50)

    buf_pdf = BytesIO()
    pdf.output(buf_pdf)
    return buf_pdf.getvalue()
