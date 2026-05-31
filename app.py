from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import joblib
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from config import (
    ARQUIVO_BASE,
    ARQUIVO_MODELO_RF,
    ARQUIVO_RESULTADO_COMPARACAO,
)
from executar_pipeline import executar_pipeline
from features_series import obter_colunas_serie
from gerar_base import gerar_base_sintetica
from gerar_relatorio_pdf import gerar_relatorio_pdf

load_dotenv(override=True)

COR_SHAP_RF   = "#f59e0b"   # âmbar  — SHAP do Random Forest
COR_LIME_RF   = "#7c3aed"   # roxo   — LIME do Random Forest
COR_LLM_BRUTO   = "#ec4899"   # pink
COR_LLM_EST     = "#16a34a"   # verde
COR_LLM_GLOBAL  = "#0284c7"   # azul céu

COR_SHAP = COR_SHAP_RF
COR_LIME = COR_LIME_RF

st.set_page_config(page_title="Modelos + Explicabilidade", layout="wide")

st.markdown("""
<style>
  .main .block-container { font-size: 0.82rem; padding-top: 1rem; }
  section[data-testid="stSidebar"] { font-size: 0.80rem; }
  h1 { font-size: 1.35rem !important; margin-bottom: 0.3rem; }
  h2 { font-size: 1.05rem !important; }
  h3 { font-size: 0.92rem !important; }
  p  { font-size: 0.82rem; }
  .stMarkdown p { font-size: 0.82rem; }
  [data-testid="stMetricLabel"]  { font-size: 0.70rem !important; }
  [data-testid="stMetricValue"]  { font-size: 1.00rem !important; }
  .stAlert { font-size: 0.78rem; }
  .stTabs [data-baseweb="tab"]  { font-size: 0.82rem; padding: 6px 14px; }
  .stDataFrame { font-size: 0.78rem; }
  .stCaption  { font-size: 0.72rem; }
</style>
""", unsafe_allow_html=True)

st.title("Modelos preditivos e explicabilidade com SHAP, LIME e LLM")

st.markdown("""
O projeto treina o Random Forest com séries temporais sintéticas e compara **quatro formas de explicabilidade** por ponto:

- **SHAP**: contribuição local de cada ponto para a decisão do modelo.
- **LIME**: aproximação local da decisão do modelo.
- **LLM bruto**: o LLM recebe os valores diários completos e aponta os pontos usados como evidência.
- **LLM estat.**: o LLM recebe estatísticas resumidas por quartil (sem os valores brutos) e faz a mesma tarefa.
""")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def carregar_json(valor, padrao):
    if valor is None or pd.isna(valor):
        return padrao
    if isinstance(valor, (list, dict)):
        return valor
    try:
        return json.loads(valor)
    except Exception:
        return padrao


def formatar_lista_legivel(valor) -> str:
    dados = carregar_json(valor, [])
    if isinstance(dados, list):
        return ", ".join(str(x) for x in dados) if dados else "Nenhum"
    return str(dados)


def carregar_metricas_dos_modelos() -> dict | None:
    metricas = {}
    if ARQUIVO_MODELO_RF.exists():
        try:
            metricas["random_forest"] = joblib.load(ARQUIVO_MODELO_RF)
        except Exception:
            pass
    return metricas if metricas else None


# ─── Pontos de explicabilidade ────────────────────────────────────────────────

_MAPA_COL = {
    "SHAP-RF":     "shap_pontos",
    "LIME-RF":     "lime_pontos",
    "LLM bruto":   "llm_bruto_pontos",
    "LLM estat.":  "llm_est_pontos",
    "LLM global":  "llm_global_pontos",
}
_MAPA_PREFIXO = {
    "SHAP-RF":     "SR",
    "LIME-RF":     "LR",
    "LLM bruto":   "B",
    "LLM estat.":  "E",
    "LLM global":  "G",
}


def preparar_pontos_metodo(linha: pd.Series, metodo: str, top_n: int) -> pd.DataFrame:
    col_chave = _MAPA_COL[metodo]
    pontos = carregar_json(linha.get(col_chave, "[]"), [])
    if not pontos:
        return pd.DataFrame()

    df = pd.DataFrame(pontos)
    if df.empty:
        return df

    df["metodo"] = metodo
    df["valor"]  = pd.to_numeric(df.get("valor"), errors="coerce")
    df["dia"]    = pd.to_numeric(df.get("dia"),   errors="coerce")
    df["contribuicao"]     = pd.to_numeric(df.get("contribuicao"), errors="coerce").fillna(0.0)
    df["abs_contribuicao"] = df["contribuicao"].abs()
    df["sentido"] = df["contribuicao"].apply(lambda v: "favorece" if v >= 0 else "reduz")

    prefixo = _MAPA_PREFIXO[metodo]
    df = df.sort_values("abs_contribuicao", ascending=False).head(top_n).copy()
    df["ranking"]     = range(1, len(df) + 1)
    df["label_curta"] = df.apply(lambda r: f"{prefixo}{int(r['ranking'])}", axis=1)

    for col in ["data", "motivo_associado", "justificativa"]:
        if col not in df.columns:
            df[col] = ""

    return df


def _dias(df: pd.DataFrame) -> set[int]:
    if df.empty or "dia" not in df.columns:
        return set()
    return set(int(x) for x in df["dia"].dropna())


# ─── Gráfico Altair ───────────────────────────────────────────────────────────

_DOMINIO_METODOS = ["SHAP-RF", "LIME-RF", "LLM bruto", "LLM estat.", "LLM global"]
_CORES_METODOS   = [COR_SHAP_RF, COR_LIME_RF, COR_LLM_BRUTO, COR_LLM_EST, COR_LLM_GLOBAL]


def construir_grafico_explicabilidade(df_serie: pd.DataFrame, df_pontos: pd.DataFrame):
    base = alt.Chart(df_serie).encode(x=alt.X("dia:Q", title="Dia"))

    linha_serie = base.mark_line(strokeWidth=2).encode(
        y=alt.Y("valor:Q", title="Valor"),
        tooltip=[alt.Tooltip("dia:Q", title="Dia"), alt.Tooltip("valor:Q", title="Valor")],
    )
    linha_media = base.mark_line(strokeDash=[6, 4], opacity=0.55).encode(
        y=alt.Y("media:Q"),
        tooltip=[alt.Tooltip("media:Q", title="Média", format=".2f")],
    )
    camadas = [linha_serie, linha_media]

    if not df_pontos.empty:
        pontos = alt.Chart(df_pontos).mark_circle(stroke="black", strokeWidth=0.5).encode(
            x=alt.X("dia:Q"),
            y=alt.Y("valor:Q"),
            color=alt.Color(
                "metodo:N",
                scale=alt.Scale(domain=_DOMINIO_METODOS, range=_CORES_METODOS),
                legend=alt.Legend(orient="top"),
            ),
            opacity=alt.Opacity(
                "abs_contribuicao:Q",
                scale=alt.Scale(range=[0.15, 1.0]),
                legend=None,
            ),
            size=alt.Size("abs_contribuicao:Q", scale=alt.Scale(range=[100, 650])),
            tooltip=[
                alt.Tooltip("metodo:N",           title="Método"),
                alt.Tooltip("ranking:Q",          title="Ranking"),
                alt.Tooltip("dia:Q",              title="Dia"),
                alt.Tooltip("data:N",             title="Data"),
                alt.Tooltip("valor:Q",            title="Valor",       format=".2f"),
                alt.Tooltip("contribuicao:Q",     title="Contribuição", format=".4f"),
                alt.Tooltip("sentido:N",          title="Efeito"),
                alt.Tooltip("motivo_associado:N", title="Motivo LLM"),
                alt.Tooltip("justificativa:N",    title="Justif. LLM"),
            ],
        )
        textos = alt.Chart(df_pontos).mark_text(dy=-14, fontSize=9, fontWeight="bold").encode(
            x="dia:Q", y="valor:Q",
            text="label_curta:N",
            color=alt.Color(
                "metodo:N",
                scale=alt.Scale(domain=_DOMINIO_METODOS, range=_CORES_METODOS),
                legend=None,
            ),
        )
        camadas.extend([pontos, textos])

    return alt.layer(*camadas).resolve_scale(size="independent").properties(height=390)


# ─── Textos didáticos ─────────────────────────────────────────────────────────

def obter_resumo_didatico(df_metodo: pd.DataFrame, metodo: str, classe_alg: str) -> str:
    if df_metodo.empty:
        return f"O {metodo} não encontrou pontos relevantes."

    principal = df_metodo.iloc[0]
    pos = df_metodo[df_metodo["contribuicao"] >= 0]
    neg = df_metodo[df_metodo["contribuicao"] <  0]

    if "LLM" in metodo:
        tipo = "bruto" if "bruto" in metodo else "estatístico"
        return (
            f"O **LLM ({tipo})** destacou **{len(df_metodo)} ponto(s)**. "
            f"**{len(pos)}** favorecem e **{len(neg)}** reduzem a classe **{classe_alg}**. "
            f"Principal: **dia {int(principal['dia'])}** — valor **{principal['valor']:.0f}**, "
            f"contribuição **{principal['contribuicao']:+.4f}**."
        )
    return (
        f"O **{metodo}** destacou **{len(df_metodo)} ponto(s)**. "
        f"**{len(pos)}** favorecem a classe **{classe_alg}** e **{len(neg)}** reduzem. "
        f"Principal: **dia {int(principal['dia'])}** ({principal['data']}), "
        f"valor **{principal['valor']:.0f}**, contribuição **{principal['contribuicao']:+.4f}**."
    )


def gerar_explicacoes_pontos(df_metodo: pd.DataFrame, metodo: str, classe_alg: str, limite: int = 5) -> list[str]:
    if df_metodo.empty:
        return [f"Nenhum ponto relevante de {metodo} encontrado."]

    explicacoes = []
    for _, p in df_metodo.head(limite).iterrows():
        efeito = "favorece" if p["contribuicao"] >= 0 else "reduz"
        pref   = _MAPA_PREFIXO[metodo]
        if "LLM" in metodo:
            just   = p.get("justificativa", "") or "Sem justificativa."
            motivo = p.get("motivo_associado", "") or "não informado"
            texto = (
                f"**{pref}{int(p['ranking'])}**: dia **{int(p['dia'])}** "
                f"({p.get('data','')}), valor **{p['valor']:.0f}**. "
                f"Efeito: **{efeito}** → **{classe_alg}** — contrib. **{p['contribuicao']:+.4f}**. "
                f"Motivo: **{motivo}**. {just}"
            )
        else:
            acao = "ajudou a manter" if p["contribuicao"] >= 0 else "puxou contra"
            texto = (
                f"**{pref}{int(p['ranking'])}**: dia **{int(p['dia'])}** "
                f"({p['data']}), valor **{p['valor']:.0f}**. "
                f"Esse ponto **{acao}** a classe **{classe_alg}**, contrib. **{p['contribuicao']:+.4f}**."
            )
        explicacoes.append(texto)
    return explicacoes


def renderizar_caixa_didatica(titulo: str, cor: str, descricao: str, resumo: str, itens: list[str]) -> None:
    st.markdown(
        f"""<div style="border-left:5px solid {cor}; padding:0.6rem 0.8rem; """
        f"""background:rgba(0,0,0,0.02); border-radius:0.4rem; margin-bottom:0.5rem;">"""
        f"""<div style="font-size:0.85rem;"><strong>{titulo}</strong></div>"""
        f"""<div style="margin-top:0.2rem; font-size:0.76rem;">{descricao}</div>"""
        f"""<div style="margin-top:0.2rem; font-size:0.76rem;">{resumo}</div></div>""",
        unsafe_allow_html=True,
    )
    for item in itens:
        st.markdown(f"<div style='font-size:0.76rem; margin-left:0.5rem;'>• {item}</div>",
                    unsafe_allow_html=True)


def renderizar_legenda_visual(metodos_ativos: list[str]) -> None:
    _COR = {
        "SHAP-RF":     COR_SHAP_RF,
        "LIME-RF":     COR_LIME_RF,
        "LLM bruto":   COR_LLM_BRUTO,
        "LLM estat.":  COR_LLM_EST,
        "LLM global":  COR_LLM_GLOBAL,
    }
    _DESC = {
        "SHAP-RF":     "SHAP — Random Forest",
        "LIME-RF":     "LIME — Random Forest",
        "LLM bruto":   "evidência — dados brutos",
        "LLM estat.":  "evidência — estatísticas por quartil",
        "LLM global":  "evidência — estatísticas globais",
    }
    itens = "".join(
        f"""<div style="padding:0.3rem 0.6rem; border:1px solid #ddd; border-radius:0.5rem; font-size:0.74rem;">
  <span style="display:inline-block;width:10px;height:10px;background:{_COR[m]};border-radius:50%;margin-right:4px;"></span>
  <strong>{m}</strong>: {_DESC[m]}
</div>"""
        for m in metodos_ativos
        if m in _COR
    )
    pref_nota = " · ".join(f"<strong>{_MAPA_PREFIXO[m]}</strong>={m}" for m in metodos_ativos if m in _MAPA_PREFIXO)
    st.markdown(
        f"""<div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin:0.3rem 0 0.6rem;">
{itens}
<div style="padding:0.3rem 0.6rem; border:1px solid #ddd; border-radius:0.5rem; font-size:0.74rem;">
  {pref_nota}
</div>
</div>""",
        unsafe_allow_html=True,
    )


# ─── Gráfico de explicabilidade ───────────────────────────────────────────────

def renderizar_grafico_explicabilidade(
    linha_rf: pd.Series | None,
    valores: list[int],
    key_prefix: str = "",
) -> None:
    """Gráfico de explicabilidade RF: SHAP-RF, LIME-RF, LLM bruto, LLM estat., LLM global."""
    st.markdown("**Explicabilidade visual por ponto da série**")

    metodos_disponiveis = ["SHAP-RF", "LIME-RF", "LLM bruto", "LLM estat.", "LLM global"]
    if linha_rf is None:
        metodos_disponiveis = [m for m in metodos_disponiveis if "RF" not in m]

    metodos_exibidos: list[str] = st.multiselect(
        "Métodos exibidos",
        metodos_disponiveis,
        default=metodos_disponiveis,
        key=f"ms_{key_prefix}",
    )

    pct_shap_lime = st.slider(
        "% dos pontos SHAP e LIME mais significativos",
        min_value=1, max_value=100, value=11, step=1,
        key=f"sld_{key_prefix}",
        help="Percentual dos dias com maior impacto a destacar para SHAP-RF e LIME-RF.",
    )

    df_serie = pd.DataFrame({
        "dia":   range(1, len(valores) + 1),
        "valor": valores,
        "media": [sum(valores) / len(valores)] * len(valores),
    })

    _linha_de = {
        "SHAP-RF":    linha_rf,
        "LIME-RF":    linha_rf,
        "LLM bruto":  linha_rf,
        "LLM estat.": linha_rf,
        "LLM global": linha_rf,
    }

    # Prepara todos os dfs (sempre, para cálculo de intersecções)
    mapa_df: dict[str, pd.DataFrame] = {}
    for metodo in metodos_disponiveis:
        fonte = _linha_de.get(metodo)
        if fonte is None:
            mapa_df[metodo] = pd.DataFrame()
        elif metodo in {"SHAP-RF", "LIME-RF"}:
            pts_raw = carregar_json(fonte.get(_MAPA_COL[metodo], "[]"), [])
            n_total = len(pts_raw) if pts_raw else len(valores)
            top_n_sl = max(1, int(pct_shap_lime / 100 * n_total))
            mapa_df[metodo] = preparar_pontos_metodo(fonte, metodo, top_n_sl)
        else:
            mapa_df[metodo] = preparar_pontos_metodo(fonte, metodo, 20)

    partes = [mapa_df[m] for m in metodos_exibidos if not mapa_df[m].empty]
    df_pontos = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()

    if df_pontos.empty:
        st.warning("Nenhum ponto para os métodos selecionados.")
        if linha_rf is not None:
            _xai = linha_rf.get("erro_xai")
            if _xai and not (isinstance(_xai, float) and pd.isna(_xai)):
                st.error(f"Erro XAI ({linha_rf.get('nome_algoritmo','')}): {_xai}")
            for pref in ("llm_bruto", "llm_est", "llm_global"):
                _v = linha_rf.get(f"{pref}_erro")
                if _v and not (isinstance(_v, float) and pd.isna(_v)):
                    st.error(f"Erro {pref}: {_v}")
        return

    renderizar_legenda_visual(metodos_exibidos)
    st.altair_chart(construir_grafico_explicabilidade(df_serie, df_pontos), use_container_width=True)

    # Métricas de contagem por método
    dias_por_metodo = {m: _dias(mapa_df[m]) for m in metodos_disponiveis}
    colunas_metricas = [(m, len(mapa_df[m])) for m in metodos_exibidos]

    # "Comuns" = interseção de todos os métodos exibidos com dados
    metodos_com_dados = [m for m in metodos_exibidos if not mapa_df[m].empty]
    if metodos_com_dados:
        todos_os_dias = set.intersection(*(dias_por_metodo[m] for m in metodos_com_dados))
        colunas_metricas.append(("Comuns", len(todos_os_dias)))
    else:
        todos_os_dias = set()

    cols = st.columns(len(colunas_metricas))
    for col, (nome, val) in zip(cols, colunas_metricas):
        col.metric(nome, val)

    st.caption(
        "SR=SHAP-RF · LR=LIME-RF · B=LLM bruto · E=LLM estat. · G=LLM global · "
        "Número = ranking por magnitude. Tamanho = magnitude."
    )

    # Tabela de intersecção (quando ≥ 2 métodos com dados)
    if len(metodos_com_dados) >= 2:
        comparacoes = []
        for i, m1 in enumerate(metodos_com_dados):
            for m2 in metodos_com_dados[i + 1:]:
                intersec = sorted(dias_por_metodo[m1] & dias_por_metodo[m2])
                comparacoes.append({
                    "Comparação": f"{m1} ∩ {m2}",
                    "Qtd": len(intersec),
                    "Dias": ", ".join(map(str, intersec)) or "—",
                })
        if len(metodos_com_dados) > 2:
            comparacoes.append({
                "Comparação": "∩ todos",
                "Qtd": len(todos_os_dias),
                "Dias": ", ".join(map(str, sorted(todos_os_dias))) or "—",
            })
        with st.expander("Sobreposição entre métodos"):
            st.dataframe(pd.DataFrame(comparacoes), use_container_width=True, hide_index=True)

    # Caixas didáticas — uma por método exibido com dados
    if metodos_com_dados:
        _COR_MAP = {
            "SHAP-RF":    (COR_SHAP_RF,    "SHAP-RF",    "Contribuição local — Random Forest."),
            "LIME-RF":    (COR_LIME_RF,    "LIME-RF",    "Aproximação local — Random Forest."),
            "LLM bruto":  (COR_LLM_BRUTO,  "LLM bruto",  "LLM com acesso aos valores diários completos."),
            "LLM estat.": (COR_LLM_EST,    "LLM estat.", "LLM com estatísticas resumidas por quartil."),
            "LLM global": (COR_LLM_GLOBAL, "LLM global", "LLM com apenas estatísticas globais da série."),
        }
        _classe_alg = {
            m: (linha_rf.get("algoritmo_previsto", "") if linha_rf is not None else "")
            for m in metodos_com_dados
        }

        cols_did = st.columns(len(metodos_com_dados))
        for col_ui, metodo in zip(cols_did, metodos_com_dados):
            with col_ui:
                cor, titulo, desc = _COR_MAP[metodo]
                renderizar_caixa_didatica(
                    titulo, cor, desc,
                    obter_resumo_didatico(mapa_df[metodo], metodo, _classe_alg[metodo]),
                    gerar_explicacoes_pontos(mapa_df[metodo], metodo, _classe_alg[metodo]),
                )


# ─── Detalhe unificado RF + DT ────────────────────────────────────────────────

def _badge(cor_bg: str, cor_txt: str, texto: str) -> str:
    return (
        f"<span style='background:{cor_bg};color:{cor_txt};"
        f"padding:2px 7px;border-radius:10px;font-size:0.72rem;'>{texto}</span>"
    )


def renderizar_detalhe_serie_unificado(
    linha_rf: pd.Series | None,
    valores: list[int],
    key_prefix: str,
) -> None:
    """Visão única por série: previsão RF + gráfico de explicabilidade."""
    classe_real = linha_rf.get("classe_real", "—") if linha_rf is not None else "—"

    prev_rf = linha_rf.get("algoritmo_previsto", "—") if linha_rf is not None else "N/D"
    acerto_rf = str(classe_real) == str(prev_rf)

    llm_b_prev = linha_rf.get("llm_bruto_previsto",  "") if linha_rf is not None else ""
    llm_e_prev = linha_rf.get("llm_est_previsto",    "") if linha_rf is not None else ""
    llm_g_prev = linha_rf.get("llm_global_previsto", "") if linha_rf is not None else ""

    cols_met = st.columns(5)
    cols_met[0].metric("Classe real", classe_real)
    cols_met[1].metric("RF previsto", prev_rf, delta="✓" if acerto_rf else "✗", delta_color="normal" if acerto_rf else "inverse")
    if llm_b_prev: cols_met[2].metric("LLM bruto",   llm_b_prev)
    if llm_e_prev: cols_met[3].metric("LLM estat.",  llm_e_prev)
    if llm_g_prev: cols_met[4].metric("LLM global",  llm_g_prev)

    if linha_rf is not None:
        for _pref, _label in [("llm_bruto", "LLM bruto"), ("llm_est", "LLM estat."), ("llm_global", "LLM global")]:
            _val = linha_rf.get(f"{_pref}_erro")
            _erro = "" if _val is None or (isinstance(_val, float) and pd.isna(_val)) else str(_val).strip()
            if _erro:
                st.error(f"Erro {_label}: {_erro}")

    badges = [
        _badge("#fef3c7", "#92400e", "✓ SHAP-RF"),
        _badge("#ede9fe", "#5b21b6", "✓ LIME-RF"),
        _badge("#fce7f3", "#9d174d", "✓ LLM bruto"),
        _badge("#dcfce7", "#166534", "✓ LLM estat."),
        _badge("#e0f2fe", "#0369a1", "✓ LLM global"),
    ]
    st.markdown(" ".join(badges), unsafe_allow_html=True)
    st.markdown("")

    if valores:
        renderizar_grafico_explicabilidade(linha_rf, valores, key_prefix)

    if linha_rf is not None:
        with st.expander("Motivos e alinhamento (LLM)"):
            col_b, col_e, col_g = st.columns(3)
            with col_b:
                st.markdown("**LLM bruto**")
                al = linha_rf.get("llm_bruto_alinhamento", "")
                sc = linha_rf.get("llm_bruto_score", "")
                if al:
                    st.markdown(f"Alinhamento: **{al}** (score: {sc})")
                st.markdown(f"Motivos: {formatar_lista_legivel(linha_rf.get('llm_bruto_motivos','[]'))}")
                st.markdown(f"Em comum: {formatar_lista_legivel(linha_rf.get('llm_bruto_motivos_em_comum','[]'))}")
                st.markdown(f"Ausentes: {formatar_lista_legivel(linha_rf.get('llm_bruto_motivos_ausentes','[]'))}")
            with col_e:
                st.markdown("**LLM estat.**")
                al = linha_rf.get("llm_est_alinhamento", "")
                sc = linha_rf.get("llm_est_score", "")
                if al:
                    st.markdown(f"Alinhamento: **{al}** (score: {sc})")
                st.markdown(f"Motivos: {formatar_lista_legivel(linha_rf.get('llm_est_motivos','[]'))}")
                st.markdown(f"Em comum: {formatar_lista_legivel(linha_rf.get('llm_est_motivos_em_comum','[]'))}")
                st.markdown(f"Ausentes: {formatar_lista_legivel(linha_rf.get('llm_est_motivos_ausentes','[]'))}")
            with col_g:
                st.markdown("**LLM global**")
                al = linha_rf.get("llm_global_alinhamento", "")
                sc = linha_rf.get("llm_global_score", "")
                if al:
                    st.markdown(f"Alinhamento: **{al}** (score: {sc})")
                st.markdown(f"Motivos: {formatar_lista_legivel(linha_rf.get('llm_global_motivos','[]'))}")
                st.markdown(f"Em comum: {formatar_lista_legivel(linha_rf.get('llm_global_motivos_em_comum','[]'))}")
                st.markdown(f"Ausentes: {formatar_lista_legivel(linha_rf.get('llm_global_motivos_ausentes','[]'))}")
            st.markdown(f"**Motivos especialista:** {formatar_lista_legivel(linha_rf.get('motivos_especialista','[]'))}")

        with st.expander("Explicação textual"):
            st.markdown(f"**Ground truth:** {linha_rf.get('explicacao_ground_truth','')}")
            col_b2, col_e2, col_g2 = st.columns(3)
            with col_b2:
                st.markdown("**LLM bruto**")
                analise_b = linha_rf.get("llm_bruto_analise", "")
                if analise_b:
                    st.markdown(f"*Análise:* {analise_b}")
                exp_b = linha_rf.get("llm_bruto_explicacao", "")
                if exp_b:
                    st.markdown(f"*Explicação:* {exp_b}")
            with col_e2:
                st.markdown("**LLM estat.**")
                analise_e = linha_rf.get("llm_est_analise", "")
                if analise_e:
                    st.markdown(f"*Análise:* {analise_e}")
                exp_e = linha_rf.get("llm_est_explicacao", "")
                if exp_e:
                    st.markdown(f"*Explicação:* {exp_e}")
            with col_g2:
                st.markdown("**LLM global**")
                analise_g = linha_rf.get("llm_global_analise", "")
                if analise_g:
                    st.markdown(f"*Análise:* {analise_g}")
                exp_g = linha_rf.get("llm_global_explicacao", "")
                if exp_g:
                    st.markdown(f"*Explicação:* {exp_g}")

        with st.expander("Probabilidades e pontos LLM"):
            st.markdown(f"**Probabilidades (RF):** {linha_rf.get('probabilidades_algoritmo','{}')}")
            col_pb, col_pe, col_pg = st.columns(3)
            with col_pb:
                st.markdown("**Pontos LLM bruto**")
                pts_b = carregar_json(linha_rf.get("llm_bruto_pontos", "[]"), [])
                if pts_b:
                    st.dataframe(pd.DataFrame(pts_b), use_container_width=True, hide_index=True)
                else:
                    st.caption("Sem pontos.")
            with col_pe:
                st.markdown("**Pontos LLM estat.**")
                pts_e = carregar_json(linha_rf.get("llm_est_pontos", "[]"), [])
                if pts_e:
                    st.dataframe(pd.DataFrame(pts_e), use_container_width=True, hide_index=True)
                else:
                    st.caption("Sem pontos.")
            with col_pg:
                st.markdown("**Pontos LLM global**")
                pts_g = carregar_json(linha_rf.get("llm_global_pontos", "[]"), [])
                if pts_g:
                    st.dataframe(pd.DataFrame(pts_g), use_container_width=True, hide_index=True)
                else:
                    st.caption("Sem pontos.")


# ─── Aba Métricas ─────────────────────────────────────────────────────────────

def renderizar_aba_metricas(metricas: dict | None) -> None:
    if metricas is None:
        st.info("Execute o pipeline para gerar as métricas dos modelos.")
        return

    melhor_chave = max(metricas, key=lambda k: metricas[k].get("accuracy", 0))
    cols = st.columns(len(metricas))
    for i, (chave, pacote) in enumerate(metricas.items()):
        with cols[i]:
            eh_melhor = chave == melhor_chave
            borda = "#22c55e" if eh_melhor else "#e5e7eb"
            acc  = pacote.get("accuracy",      0.0)
            f1   = pacote.get("f1_weighted",   0.0)
            prec = pacote.get("precisao_macro", 0.0)
            rec  = pacote.get("recall_macro",  0.0)
            nome = pacote.get("nome_algoritmo", chave)
            tag  = " ★" if eh_melhor else ""
            st.markdown(
                f"""<div style="border:2px solid {borda};border-radius:8px;padding:10px;text-align:center;">
  <div style="font-size:0.72rem;color:#555;font-weight:600;">{nome}{tag}</div>
  <div style="font-size:1.55rem;font-weight:700;margin:4px 0;">{acc:.1%}</div>
  <div style="font-size:0.68rem;color:#888;">acurácia</div>
  <div style="margin-top:6px;font-size:0.70rem;color:#444;">
    F1 {f1:.3f} · P {prec:.3f} · R {rec:.3f}
  </div>
</div>""",
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.markdown("**Tabela comparativa**")
    linhas = [
        {
            "Algoritmo":        p.get("nome_algoritmo", k),
            "Acurácia":         p.get("accuracy",       0.0),
            "F1 (Weighted)":    p.get("f1_weighted",    0.0),
            "Precisão (Macro)": p.get("precisao_macro", 0.0),
            "Recall (Macro)":   p.get("recall_macro",   0.0),
        }
        for k, p in metricas.items()
    ]
    df_m = pd.DataFrame(linhas).set_index("Algoritmo")
    try:
        styled = df_m.style.format("{:.4f}").highlight_max(
            axis=0, props="background-color:#dcfce7; color:#166534; font-weight:bold;"
        )
        st.dataframe(styled, use_container_width=True)
    except Exception:
        st.dataframe(df_m.round(4), use_container_width=True)

    st.markdown("**Comparação visual**")
    st.bar_chart(df_m, use_container_width=True)

    with st.expander("O que significa cada métrica?"):
        st.markdown("""
- **Acurácia**: proporção de previsões corretas.
- **F1 (Weighted)**: média harmônica entre precisão e recall, ponderada pelo tamanho de cada classe.
- **Precisão (Macro)**: dos alunos classificados em cada grupo, quantos realmente pertencem a ele. Média entre classes.
- **Recall (Macro)**: dos alunos que pertencem a cada grupo, quantos foram corretamente identificados. Média entre classes.
""")


# ─── Aba Relatório ────────────────────────────────────────────────────────────

_METODOS_RELATORIO  = ["SHAP-RF", "LIME-RF", "LLM bruto", "LLM estat.", "LLM global"]
_METODOS_COM_CONTRIB = {"SHAP-RF", "LIME-RF"}
_LABELS_RELATORIO   = {
    "SHAP-RF":    "SHAP - RF",
    "LIME-RF":    "LIME - RF",
    "LLM bruto":  "LLM Bruto",
    "LLM estat.": "LLM Estat.",
    "LLM global": "LLM Global",
}


def renderizar_aba_relatorio(df_resultados: pd.DataFrame | None) -> None:
    if df_resultados is None or df_resultados.empty:
        st.info("Execute o pipeline primeiro.")
        return

    df_unicas = df_resultados.drop_duplicates("id_serie")

    # Cada entrada: (id_serie, vals, {metodo: [{"dia": int, "abs_contrib": float}]})
    dados_a: list[tuple] = []
    dados_b: list[tuple] = []

    for _, row in df_unicas.iterrows():
        vals = carregar_json(row.get("serie_temporal", "[]"), [])
        if not vals:
            continue
        classe = str(row.get("classe_real", ""))
        pts_metodo: dict[str, list[dict]] = {}
        for metodo in _METODOS_RELATORIO:
            col = _MAPA_COL.get(metodo, "")
            pts = carregar_json(row.get(col, "[]"), [])
            pontos_info = []
            for p in pts:
                d = p.get("dia")
                if d is None:
                    continue
                c = p.get("contribuicao")
                if c is None:
                    c = p.get("importancia", 0)
                try:
                    abs_c = abs(float(c)) if c is not None else 0.0
                except (TypeError, ValueError):
                    abs_c = 0.0
                try:
                    pontos_info.append({"dia": int(d), "abs_contrib": abs_c})
                except (TypeError, ValueError):
                    pass
            pts_metodo[metodo] = pontos_info
        entrada = (str(row.get("id_serie", "")), vals, pts_metodo)
        if classe == "perfil_A":
            dados_a.append(entrada)
        elif classe == "perfil_B":
            dados_b.append(entrada)

    if not dados_a and not dados_b:
        st.warning("Nenhuma série encontrada nos resultados.")
        return

    # ── Escala Y global ───────────────────────────────────────────────────────
    todas_vals = [v for (_, vals, _) in dados_a + dados_b for v in vals]
    y_min = min(todas_vals)
    y_max = max(todas_vals)
    margem = (y_max - y_min) * 0.05
    y_min -= margem
    y_max += margem

    # ── Slider % SHAP/LIME ────────────────────────────────────────────────────
    pct_shap_lime_rel = st.slider(
        "% dos pontos SHAP e LIME mais significativos",
        min_value=1, max_value=100, value=11, step=1,
        key="rel_pct_shap_lime",
        help="Percentual dos dias com maior impacto absoluto a destacar nos gráficos SHAP-RF e LIME-RF.",
    )

    # ── Função de plotagem ────────────────────────────────────────────────────
    def _plotar_perfil(dados: list, titulo: str) -> None:
        if not dados:
            st.info(f"Nenhuma série de {titulo} encontrada.")
            return

        n    = len(dados)
        cmap = plt.colormaps["tab20" if n > 10 else "tab10"].resampled(max(n, 1))
        cores = [cmap(i) for i in range(n)]

        st.subheader(titulo)
        fig, axes = plt.subplots(5, 1, figsize=(13, 16), sharex=True, sharey=True)

        for ax, metodo in zip(axes, _METODOS_RELATORIO):
            contribs_metodo: list[float] = []
            for i, (id_serie, vals, pts_metodo) in enumerate(dados):
                cor  = cores[i]
                dias = list(range(1, len(vals) + 1))
                ax.plot(dias, vals, color=cor, linewidth=0.9, alpha=0.55, label=id_serie)

                pts = pts_metodo.get(metodo, [])
                if metodo in _METODOS_COM_CONTRIB and pts:
                    pts_sorted = sorted(pts, key=lambda p: p["abs_contrib"], reverse=True)
                    top_k = max(1, int(pct_shap_lime_rel / 100 * len(pts_sorted)))
                    pts = pts_sorted[:top_k]

                    # alpha proporcional ao impacto dentro do conjunto filtrado
                    c_vals = [p["abs_contrib"] for p in pts]
                    c_lo, c_hi = min(c_vals), max(c_vals)
                    c_rng = c_hi - c_lo if c_hi > c_lo else None
                    r, g, b, _ = cor

                    for p in pts:
                        d = p["dia"]
                        if not (1 <= d <= len(vals)):
                            continue
                        alpha = 1.0 if c_rng is None else 0.2 + 0.8 * (p["abs_contrib"] - c_lo) / c_rng
                        ax.scatter(
                            d, vals[d - 1],
                            color=(float(r), float(g), float(b)), alpha=float(alpha),
                            marker="x", s=70, linewidths=1.6, zorder=4,
                        )
                        contribs_metodo.append(p["abs_contrib"])
                else:
                    for p in pts:
                        d = p["dia"]
                        if not (1 <= d <= len(vals)):
                            continue
                        r2, g2, b2, a2 = cor
                        ax.scatter(
                            d, vals[d - 1],
                            color=(float(r2), float(g2), float(b2)), alpha=float(a2),
                            marker="x", s=70, linewidths=1.6, zorder=4,
                        )

            ax.set_ylabel(_LABELS_RELATORIO[metodo], fontsize=8, labelpad=4)
            ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.2, linewidth=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=6)

            if metodo in _METODOS_COM_CONTRIB and contribs_metodo:
                c_min = min(contribs_metodo)
                c_max = max(contribs_metodo)
                ax.text(
                    0.995, 0.97,
                    f"impacto  mín={c_min:.4f}  máx={c_max:.4f}",
                    transform=ax.transAxes, fontsize=6.5, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="#cccccc"),
                )

        axes[-1].set_xlabel("Dia", fontsize=8)

        handles, labels = axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[0].legend(
            by_label.values(), by_label.keys(),
            fontsize=6, loc="upper right",
            ncol=max(1, n // 5),
            framealpha=0.8,
        )

        fig.suptitle(titulo, fontsize=11, y=1.002)
        fig.tight_layout(pad=0.5, h_pad=0.5)
        st.pyplot(fig)
        plt.close(fig)

    _plotar_perfil(dados_a, f"Conjunto de Teste — Perfil A  ({len(dados_a)} séries)")
    _plotar_perfil(dados_b, f"Conjunto de Teste — Perfil B  ({len(dados_b)} séries)")


# ─── Aba Métricas XAI ─────────────────────────────────────────────────────────

_QUARTIS = {
    "Q1 (1–45)":    (1,   45),
    "Q2 (46–90)":   (46,  90),
    "Q3 (91–135)":  (91,  135),
    "Q4 (136–181)": (136, 181),
}
_EVENTOS = ["Picos extremos", "Vales / outliers", "Rupturas abruptas"]


def _dias_selecionados_metodo(row: pd.Series, metodo: str) -> set[int]:
    col = _MAPA_COL.get(metodo, "")
    pontos = carregar_json(row.get(col, "[]"), [])
    dias: set[int] = set()
    for p in pontos:
        d = p.get("dia")
        if d is not None:
            dias.add(int(d))
    return dias


def renderizar_aba_metricas_xai(df_resultados: pd.DataFrame | None) -> None:
    if df_resultados is None or df_resultados.empty:
        st.info("Execute o pipeline primeiro.")
        return

    METODOS = list(_MAPA_COL.keys())
    df_unicas = df_resultados.drop_duplicates("id_serie")

    # ── Métrica 1: cobertura de quartis ───────────────────────────────────────
    contagem = {m: {q: 0 for q in _QUARTIS} for m in METODOS}
    total_pts = {m: 0 for m in METODOS}

    for _, row in df_unicas.iterrows():
        for metodo in METODOS:
            for dia in _dias_selecionados_metodo(row, metodo):
                total_pts[metodo] += 1
                for nome_q, (ini, fim) in _QUARTIS.items():
                    if ini <= dia <= fim:
                        contagem[metodo][nome_q] += 1
                        break

    prop = {
        m: {q: (contagem[m][q] / total_pts[m]) if total_pts[m] > 0 else 0.0 for q in _QUARTIS}
        for m in METODOS
    }
    df_quartis = pd.DataFrame(prop, index=list(_QUARTIS.keys())).T  # métodos × quartis

    st.subheader("1. Cobertura de quartis")
    st.caption(
        "Proporção dos pontos selecionados por cada método em cada quartil temporal. "
        "Uma distribuição uniforme (~25% por quartil) sugere ausência de viés temporal."
    )

    fig_q, ax_q = plt.subplots(figsize=(9, 3.5))
    data_q = df_quartis.values.astype(float)
    im = ax_q.imshow(data_q, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax_q.set_xticks(range(len(_QUARTIS)))
    ax_q.set_xticklabels(list(_QUARTIS.keys()), fontsize=9)
    ax_q.set_yticks(range(len(METODOS)))
    ax_q.set_yticklabels(METODOS, fontsize=9)
    for i in range(len(METODOS)):
        for j in range(len(_QUARTIS)):
            v = data_q[i, j]
            ax_q.text(j, i, f"{v:.0%}", ha="center", va="center",
                      fontsize=9, color="white" if v > 0.55 else "black")
    plt.colorbar(im, ax=ax_q, label="Proporção")
    plt.tight_layout()
    st.pyplot(fig_q)
    plt.close(fig_q)

    try:
        st.dataframe(
            df_quartis.style.format("{:.1%}").background_gradient(cmap="YlOrRd", axis=None),
            use_container_width=True,
        )
    except Exception:
        st.dataframe(df_quartis.map(lambda x: f"{x:.1%}"), use_container_width=True)

    # ── Métrica 2: taxa de detecção de eventos estruturais ────────────────────
    st.subheader("2. Taxa de detecção de eventos estruturais")
    st.caption(
        "Fração dos eventos estruturais de cada série que foram cobertos por pelo menos um ponto "
        "selecionado pelo método. Limiar: média ± 1,5 × desvio-padrão (picos/vales); "
        "1,5 × dp das diferenças consecutivas (rupturas)."
    )

    total_ev = {e: 0 for e in _EVENTOS}
    detec = {m: {e: 0 for e in _EVENTOS} for m in METODOS}

    for _, row in df_unicas.iterrows():
        serie = carregar_json(row.get("serie_temporal", "[]"), [])
        if not serie or len(serie) < 2:
            continue

        vals = np.array(serie, dtype=float)
        media = vals.mean()
        std   = vals.std()

        dias_pico = {i + 1 for i, v in enumerate(vals) if v > media + 1.5 * std}
        dias_vale = {i + 1 for i, v in enumerate(vals) if v < media - 1.5 * std}

        diffs = np.abs(np.diff(vals))
        std_diffs = diffs.std()
        limiar_rup = 1.5 * std_diffs if std_diffs > 0 else float("inf")
        dias_rup = {i + 2 for i, d in enumerate(diffs) if d > limiar_rup}

        mapa_ev = {
            "Picos extremos":    dias_pico,
            "Vales / outliers":  dias_vale,
            "Rupturas abruptas": dias_rup,
        }

        for ev, dias_ev in mapa_ev.items():
            total_ev[ev] += len(dias_ev)

        for metodo in METODOS:
            dias_sel = _dias_selecionados_metodo(row, metodo)
            for ev, dias_ev in mapa_ev.items():
                detec[metodo][ev] += len(dias_ev & dias_sel)

    linhas = []
    for ev in _EVENTOS:
        tot = total_ev[ev]
        linha: dict = {"Evento": ev, "Total eventos": tot}
        for metodo in METODOS:
            linha[metodo] = detec[metodo][ev] / tot if tot > 0 else 0.0
        linhas.append(linha)

    df_ev = pd.DataFrame(linhas).set_index("Evento")

    colunas_pct = [m for m in METODOS if m in df_ev.columns]
    try:
        styled = (
            df_ev.style
            .format({m: "{:.1%}" for m in colunas_pct})
            .format({"Total eventos": "{:d}"})
            .background_gradient(cmap="Blues", subset=colunas_pct, axis=None)
        )
        st.dataframe(styled, use_container_width=True)
    except Exception:
        st.dataframe(df_ev, use_container_width=True)

    # ── Métrica 3: Índice de Priorização de Eventos (IPE) ─────────────────────
    st.subheader("3. Índice de Priorização de Eventos (IPE)")
    st.caption(
        "Mede o quanto cada método concentra atenção nos eventos estruturais em relação ao acaso. "
        "Para todos os métodos: IPE = (fração dos pontos selecionados que são eventos) / "
        "(fração de dias de evento na série). IPE > 1 → prioriza eventos; IPE < 1 → subestima eventos."
    )

    METODOS_CONTRIB = ["SHAP-RF", "LIME-RF"]
    ipe_sel_soma = {m: {e: 0.0 for e in _EVENTOS} for m in METODOS}
    ipe_sel_n    = {m: {e: 0   for e in _EVENTOS} for m in METODOS}
    ipe_imp_soma = {m: {e: 0.0 for e in _EVENTOS} for m in METODOS_CONTRIB}
    ipe_imp_n    = {m: {e: 0   for e in _EVENTOS} for m in METODOS_CONTRIB}

    for _, row in df_unicas.iterrows():
        serie = carregar_json(row.get("serie_temporal", "[]"), [])
        if not serie or len(serie) < 2:
            continue

        vals  = np.array(serie, dtype=float)
        media = vals.mean()
        std   = vals.std()

        dias_pico = {i + 1 for i, v in enumerate(vals) if v > media + 1.5 * std}
        dias_vale = {i + 1 for i, v in enumerate(vals) if v < media - 1.5 * std}
        diffs     = np.abs(np.diff(vals))
        std_diffs = diffs.std()
        limiar_rup = 1.5 * std_diffs if std_diffs > 0 else float("inf")
        dias_rup  = {i + 2 for i, d in enumerate(diffs) if d > limiar_rup}

        mapa_ev_ipe = {
            "Picos extremos":    dias_pico,
            "Vales / outliers":  dias_vale,
            "Rupturas abruptas": dias_rup,
        }

        for metodo in METODOS:
            col = _MAPA_COL.get(metodo, "")
            pontos = carregar_json(row.get(col, "[]"), [])
            if not pontos:
                continue

            dias_contrib: dict[int, float] = {}
            for p in pontos:
                d = p.get("dia")
                c = p.get("contribuicao")
                if d is not None:
                    dias_contrib[int(d)] = abs(float(c)) if c is not None else 0.0

            n_sel = len(dias_contrib)
            if n_sel == 0:
                continue

            for ev, dias_ev in mapa_ev_ipe.items():
                n_ev = len(dias_ev)
                if n_ev == 0:
                    continue

                frac_base  = n_ev / len(vals)
                n_sel_ev   = sum(1 for d in dias_contrib if d in dias_ev)
                frac_sel   = n_sel_ev / n_sel
                ipe        = frac_sel / frac_base
                ipe_sel_soma[metodo][ev] += ipe
                ipe_sel_n[metodo][ev]   += 1

                if metodo in METODOS_CONTRIB:
                    cv_ev     = [dias_contrib[d] for d in dias_contrib if d in dias_ev]
                    cv_nao_ev = [dias_contrib[d] for d in dias_contrib if d not in dias_ev]
                    if cv_ev and cv_nao_ev:
                        ipe_imp_soma[metodo][ev] += np.mean(cv_ev) / np.mean(cv_nao_ev)
                        ipe_imp_n[metodo][ev]    += 1

    df_ipe_sel = pd.DataFrame(
        {
            m: {
                ev: ipe_sel_soma[m][ev] / ipe_sel_n[m][ev] if ipe_sel_n[m][ev] > 0 else np.nan
                for ev in _EVENTOS
            }
            for m in METODOS
        }
    ).T

    fig_ipe, ax_ipe = plt.subplots(figsize=(9, 3.5))
    data_ipe = df_ipe_sel.values.astype(float)
    vmax_ipe = max(float(np.nanmax(data_ipe)) if not np.all(np.isnan(data_ipe)) else 2.0, 2.0)
    im_ipe = ax_ipe.imshow(data_ipe, aspect="auto", cmap="RdYlGn", vmin=0, vmax=vmax_ipe)
    ax_ipe.set_xticks(range(len(_EVENTOS)))
    ax_ipe.set_xticklabels(_EVENTOS, fontsize=9)
    ax_ipe.set_yticks(range(len(METODOS)))
    ax_ipe.set_yticklabels(METODOS, fontsize=9)
    for i in range(len(METODOS)):
        for j in range(len(_EVENTOS)):
            v = data_ipe[i, j]
            if not np.isnan(v):
                ax_ipe.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="black")
    plt.colorbar(im_ipe, ax=ax_ipe, label="IPE (1,0 = acaso)")
    plt.tight_layout()
    st.pyplot(fig_ipe)
    plt.close(fig_ipe)

    st.caption("Verde = prioriza eventos (IPE > 1)  ·  Vermelho = subestima (IPE < 1)  ·  Referência: 1,0 = acaso")

    st.markdown("**IPE de importância — SHAP-RF e LIME-RF**")
    st.caption(
        "Razão entre a contribuição média (|valor|) nos dias de evento e nos dias sem evento, "
        "entre os pontos selecionados pelo método. "
        "IPE < 1 indica que o modelo atribui menor importância aos eventos do que ao restante da série."
    )
    linhas_imp = []
    for ev in _EVENTOS:
        linha: dict = {"Evento": ev}
        for m in METODOS_CONTRIB:
            n = ipe_imp_n[m][ev]
            linha[m] = ipe_imp_soma[m][ev] / n if n > 0 else np.nan
        linhas_imp.append(linha)

    df_ipe_imp = pd.DataFrame(linhas_imp).set_index("Evento")
    try:
        st.dataframe(
            df_ipe_imp.style.format("{:.3f}", na_rep="—").background_gradient(cmap="RdYlGn", axis=None),
            use_container_width=True,
        )
    except Exception:
        st.dataframe(df_ipe_imp, use_container_width=True)

    # ── Métrica 4: Posição média dos eventos no ranking SHAP/LIME ─────────────
    st.subheader("4. Posição dos eventos estruturais no ranking SHAP/LIME")
    st.caption(
        "Para SHAP e LIME, todos os 181 dias recebem um ranking de importância "
        "(1 = mais importante, 181 = menos importante). A tabela mostra a posição "
        "média dos dias de evento nesse ranking. Referência: 91 = acaso puro. "
        "Valores acima de 91 indicam que o método trata esses eventos como menos "
        "importantes que a média da série."
    )

    METODOS_RANK = ["SHAP-RF", "LIME-RF"]
    METODOS_DET  = ["LLM bruto", "LLM estat.", "LLM global"]

    rank_soma = {m: {e: 0.0 for e in _EVENTOS} for m in METODOS_RANK}
    rank_n    = {m: {e: 0   for e in _EVENTOS} for m in METODOS_RANK}
    det_soma  = {m: {e: 0   for e in _EVENTOS} for m in METODOS_DET}
    det_tot   = {e: 0 for e in _EVENTOS}

    for _, row in df_unicas.iterrows():
        serie = carregar_json(row.get("serie_temporal", "[]"), [])
        if not serie or len(serie) < 2:
            continue

        vals  = np.array(serie, dtype=float)
        media = vals.mean()
        std   = vals.std()
        diffs = np.abs(np.diff(vals))
        std_diffs = diffs.std()
        limiar_rup = 1.5 * std_diffs if std_diffs > 0 else float("inf")

        mapa_ev4 = {
            "Picos extremos":    {i + 1 for i, v in enumerate(vals) if v > media + 1.5 * std},
            "Vales / outliers":  {i + 1 for i, v in enumerate(vals) if v < media - 1.5 * std},
            "Rupturas abruptas": {i + 2 for i, d in enumerate(diffs) if d > limiar_rup},
        }

        for ev, dias_ev in mapa_ev4.items():
            det_tot[ev] += len(dias_ev)

        for metodo in METODOS_RANK:
            col = _MAPA_COL.get(metodo, "")
            pontos = carregar_json(row.get(col, "[]"), [])
            dia_para_rank: dict[int, int] = {}
            for p in pontos:
                d = p.get("dia")
                r = p.get("ranking")
                if d is not None and r is not None:
                    try:
                        dia_para_rank[int(d)] = int(r)
                    except (TypeError, ValueError):
                        pass
            for ev, dias_ev in mapa_ev4.items():
                for dia in dias_ev:
                    if dia in dia_para_rank:
                        rank_soma[metodo][ev] += dia_para_rank[dia]
                        rank_n[metodo][ev]    += 1

        for metodo in METODOS_DET:
            dias_sel = _dias_selecionados_metodo(row, metodo)
            for ev, dias_ev in mapa_ev4.items():
                det_soma[metodo][ev] += len(dias_ev & dias_sel)

    linhas_rank = []
    for ev in _EVENTOS:
        linha: dict = {"Evento": ev}
        for m in METODOS_RANK:
            n = rank_n[m][ev]
            linha[m] = rank_soma[m][ev] / n if n > 0 else np.nan
        linhas_rank.append(linha)

    linhas_det = []
    for ev in _EVENTOS:
        linha = {"Evento": ev}
        for m in METODOS_DET:
            tot = det_tot[ev]
            linha[m] = det_soma[m][ev] / tot if tot > 0 else np.nan
        linhas_det.append(linha)

    df_rank = pd.DataFrame(linhas_rank).set_index("Evento")
    df_det  = pd.DataFrame(linhas_det).set_index("Evento")

    col_r, col_d = st.columns(2)

    with col_r:
        st.markdown("**SHAP / LIME — posição média dos eventos (1–181)**")
        st.caption("Referência: 91 = acaso  ·  > 91 = evento subvalorizado")
        try:
            st.dataframe(
                df_rank.style
                .format("{:.1f}", na_rep="—")
                .background_gradient(cmap="RdYlGn_r", axis=None, vmin=1, vmax=181),
                use_container_width=True,
            )
        except Exception:
            st.dataframe(df_rank, use_container_width=True)

    with col_d:
        st.markdown("**LLMs — taxa de detecção dos eventos**")
        st.caption("% dos dias de evento selecionados pelo método")
        try:
            st.dataframe(
                df_det.style
                .format("{:.1%}", na_rep="—")
                .background_gradient(cmap="Blues", axis=None),
                use_container_width=True,
            )
        except Exception:
            st.dataframe(df_det, use_container_width=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Controles")

    st.markdown("**Configurações do LLM**")
    incluir_fi = st.checkbox(
        "Incluir feature importance no LLM",
        value=True,
        help="Envia as features mais importantes do algoritmo no prompt do LLM.",
    )
    incluir_top_regras = st.checkbox(
        "Incluir caminhos de decisão no LLM",
        value=True,
        help="Envia ao LLM as features mais frequentes nos caminhos de decisão do modelo para a série analisada.",
    )
    algoritmos_llm = ["random_forest"]

    st.divider()

    if st.button("1. Gerar base sintética", use_container_width=True):
        df_base = gerar_base_sintetica()
        st.success(f"Base criada com {len(df_base)} séries.")

    if st.button("2. Executar pipeline", use_container_width=True):
        with st.spinner("Treinando e executando pipeline…"):
            df_resultados, metricas_treino, prompts_exemplo, exemplos_por_algoritmo = executar_pipeline(
                incluir_feature_importance=incluir_fi,
                incluir_top_regras=incluir_top_regras,
                incluir_serie_bruta=True,
                algoritmos_llm=algoritmos_llm,
            )
        st.session_state["df_resultados"]         = df_resultados
        st.session_state["metricas_treino"]       = metricas_treino
        st.session_state["prompts_exemplo"]       = prompts_exemplo
        st.session_state["exemplos_por_algoritmo"] = exemplos_por_algoritmo
        st.session_state["parametros_llm"]        = {
            "incluir_feature_importance": incluir_fi,
            "incluir_top_regras":         incluir_top_regras,
            "incluir_serie_bruta":        True,
            "algoritmos_llm":             algoritmos_llm,
        }
        n_series = df_resultados["id_serie"].nunique()
        n_alg    = df_resultados["tipo_algoritmo"].nunique()
        st.success(f"Concluído — {n_series} séries × {n_alg} algoritmos.")

    st.divider()
    st.markdown("**Exportar para Claude**")

    _ARQUIVOS_CODIGO = [
        "app.py", "executar_pipeline.py", "treinar_modelos.py",
        "features_series.py", "teste_llm_modelos.py",
        "explicabilidade_pontos.py", "config.py",
        "gerar_base.py", "utils_json.py",
    ]
    _ROOT = Path(__file__).parent

    df_exp = st.session_state.get("df_resultados")
    if df_exp is None and ARQUIVO_RESULTADO_COMPARACAO.exists():
        df_exp = pd.read_csv(ARQUIVO_RESULTADO_COMPARACAO)

    # ── CSV ──────────────────────────────────────────────────────────────
    if df_exp is not None:
        csv_bytes = df_exp.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇ Exportar Resultados (CSV)",
            data=csv_bytes,
            file_name="comparacao_resultados.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.caption("Execute o pipeline para habilitar o download dos resultados.")

    # ── ZIP (resultados + código) ─────────────────────────────────────────
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if df_exp is not None:
            zf.writestr(
                "resultados/comparacao_resultados.csv",
                df_exp.to_csv(index=False),
            )
        for nome in _ARQUIVOS_CODIGO:
            caminho = _ROOT / nome
            if caminho.exists():
                zf.write(caminho, f"codigo/{nome}")
    buf.seek(0)

    st.download_button(
        "⬇ Exportar Resultados + Código (ZIP)",
        data=buf,
        file_name="estudo_xai_series_temporais.zip",
        mime="application/zip",
        use_container_width=True,
    )

    # ── PDF ───────────────────────────────────────────────────────────────────
    if df_exp is not None:
        _params_pdf   = st.session_state.get("parametros_llm", {})
        _prompts_pdf  = st.session_state.get("prompts_exemplo", {})
        _exemplos_pdf = st.session_state.get("exemplos_por_algoritmo", {})
        if st.button("Gerar Relatório PDF", use_container_width=True):
            try:
                pdf_bytes = gerar_relatorio_pdf(df_exp, _params_pdf, _prompts_pdf, _exemplos_pdf)
                st.session_state["pdf_bytes"] = pdf_bytes
            except ImportError:
                st.warning("Instale fpdf2 para habilitar o PDF: `pip install fpdf2`")
            except Exception as e:
                st.error(f"Erro ao gerar PDF: {e}")
        if "pdf_bytes" in st.session_state:
            st.download_button(
                "⬇ Relatório de teste (PDF)",
                data=st.session_state["pdf_bytes"],
                file_name="relatorio_xai.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


# ─── Tabs ─────────────────────────────────────────────────────────────────────

aba_base, aba_metricas, aba_resultados, aba_detalhe, aba_relatorio, aba_metricas_xai = st.tabs(
    ["Base", "Métricas dos modelos", "Resultados", "Detalhe por série", "Relatório", "Métricas XAI"]
)

# ── Base ──────────────────────────────────────────────────────────────────────
with aba_base:
    st.header("Base sintética")
    if not ARQUIVO_BASE.exists():
        st.info("Gere a base sintética primeiro.")
    else:
        df_base = pd.read_csv(ARQUIVO_BASE)
        st.dataframe(df_base, use_container_width=True, hide_index=True)

        colunas_serie = obter_colunas_serie(df_base)
        id_serie = st.selectbox("Visualizar série", df_base["id_serie"].astype(str).tolist(), key="base_sel")
        linha_b  = df_base[df_base["id_serie"].astype(str) == id_serie].iloc[0]
        valores  = [int(linha_b[col]) for col in colunas_serie]
        media    = sum(valores) / len(valores)
        st.line_chart(
            pd.DataFrame({"dia": range(1, len(valores) + 1), "valor": valores, "média": [media] * len(valores)}),
            x="dia", y=["valor", "média"],
        )

# ── Métricas ──────────────────────────────────────────────────────────────────
with aba_metricas:
    st.header("Métricas dos modelos")
    metricas = st.session_state.get("metricas_treino") or carregar_metricas_dos_modelos()
    renderizar_aba_metricas(metricas)

# ── Resultados ────────────────────────────────────────────────────────────────
with aba_resultados:
    st.header("Resultados do pipeline")

    df_resultados = st.session_state.get("df_resultados")
    if df_resultados is None and ARQUIVO_RESULTADO_COMPARACAO.exists():
        df_resultados = pd.read_csv(ARQUIVO_RESULTADO_COMPARACAO)

    if df_resultados is None or df_resultados.empty:
        st.info("Execute o pipeline para gerar os resultados.")
    else:
        algoritmos_disp = df_resultados["nome_algoritmo"].unique().tolist()
        filtro = st.multiselect("Filtrar por algoritmo", algoritmos_disp, default=algoritmos_disp)
        df_filt = df_resultados[df_resultados["nome_algoritmo"].isin(filtro)]

        colunas = [
            "id_serie", "classe_real", "nome_algoritmo", "algoritmo_previsto",
            "llm_bruto_previsto", "llm_bruto_alinhamento",
            "llm_est_previsto",   "llm_est_alinhamento",
            "llm_global_previsto", "llm_global_alinhamento",
            "shap_top_20", "lime_top_20",
            "llm_bruto_top_20", "llm_est_top_20", "llm_global_top_20",
            "erro_xai", "llm_bruto_erro", "llm_est_erro", "llm_global_erro",
        ]
        colunas = [c for c in colunas if c in df_filt.columns]
        st.dataframe(df_filt[colunas], use_container_width=True, hide_index=True)

        for prefixo, label in [("llm_bruto_alinhamento", "LLM bruto"), ("llm_est_alinhamento", "LLM estat."), ("llm_global_alinhamento", "LLM global")]:
            if prefixo in df_filt.columns and df_filt[prefixo].notna().any() and (df_filt[prefixo] != "").any():
                st.markdown(f"**Distribuição alinhamento {label}**")
                st.dataframe(
                    df_filt[prefixo].value_counts(dropna=False).reset_index(),
                    use_container_width=True, hide_index=True,
                )

# ── Detalhe por série ─────────────────────────────────────────────────────────
with aba_detalhe:
    st.header("Detalhe por série")

    df_resultados = st.session_state.get("df_resultados")
    if df_resultados is None and ARQUIVO_RESULTADO_COMPARACAO.exists():
        df_resultados = pd.read_csv(ARQUIVO_RESULTADO_COMPARACAO)

    if df_resultados is None or df_resultados.empty:
        st.info("Execute o pipeline primeiro.")
    else:
        ids_unicos = df_resultados["id_serie"].astype(str).unique().tolist()
        id_serie   = st.selectbox("Selecione a série", ids_unicos, key="det_sel")

        df_serie = df_resultados[df_resultados["id_serie"].astype(str) == id_serie]

        if not df_serie.empty:
            primeira = df_serie.iloc[0]
            valores = carregar_json(primeira.get("serie_temporal", "[]"), [])

            gt = primeira.get("explicacao_ground_truth", "")
            if gt:
                st.markdown(
                    f"<div style='font-size:0.78rem; margin-bottom:0.5rem;'>"
                    f"<strong>Ground truth:</strong> {gt}</div>",
                    unsafe_allow_html=True,
                )

            st.divider()

            mask_rf = df_serie["tipo_algoritmo"] == "random_forest"
            linha_rf = df_serie[mask_rf].iloc[0] if mask_rf.any() else None

            renderizar_detalhe_serie_unificado(
                linha_rf, valores,
                key_prefix=id_serie,
            )

# ── Relatório ─────────────────────────────────────────────────────────────────
with aba_relatorio:
    st.header("Relatório — padrão das séries de teste")

    df_rel = st.session_state.get("df_resultados")
    if df_rel is None and ARQUIVO_RESULTADO_COMPARACAO.exists():
        df_rel = pd.read_csv(ARQUIVO_RESULTADO_COMPARACAO)

    renderizar_aba_relatorio(df_rel)

# ── Métricas XAI ──────────────────────────────────────────────────────────────
with aba_metricas_xai:
    st.header("Métricas XAI — comparação entre métodos de explicabilidade")

    df_xai = st.session_state.get("df_resultados")
    if df_xai is None and ARQUIVO_RESULTADO_COMPARACAO.exists():
        df_xai = pd.read_csv(ARQUIVO_RESULTADO_COMPARACAO)

    renderizar_aba_metricas_xai(df_xai)
