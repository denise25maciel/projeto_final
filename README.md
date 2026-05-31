# Explicabilidade XAI em Séries Temporais Educacionais

Estudo comparativo entre **SHAP**, **LIME** e três modos de **LLM** para identificação de pontos relevantes em séries temporais de acesso a material didático.

## Objetivo

Comparar como cada método de explicabilidade seleciona pontos significativos em uma série temporal de 181 dias, sem pedir ao LLM que classifique a série. O LLM recebe apenas os dados da série e é instruído a identificar pontos estatisticamente notáveis (picos, vales, rupturas, outliers).

## Fluxo do experimento

1. Geração de base sintética com dois perfis de estudante: `perfil_A` (rotina regular) e `perfil_B` (rotina irregular).
2. Treino do **Random Forest** com as colunas `dia_*` da série temporal.
3. Cálculo de explicabilidade local para cada série do conjunto de teste:
   - **SHAP-RF** — contribuição por dia via TreeSHAP.
   - **LIME-RF** — aproximação linear local por dia.
   - **LLM bruto** — LLM recebe os valores diários completos.
   - **LLM estat.** — LLM recebe estatísticas resumidas por quartil.
   - **LLM global** — LLM recebe apenas estatísticas globais da série.
4. Comparação dos pontos selecionados por cada método usando quatro métricas XAI.

O LLM **não** é pedido para classificar a série nem para explicar a decisão do modelo — sua tarefa é exclusivamente identificar os pontos estatisticamente mais relevantes.

## Métricas XAI

### Métrica 1 — Cobertura de quartis
Proporção dos pontos selecionados por cada método que recaem em cada um dos quatro quartis temporais (Q1 = dias 1–45, Q2 = 46–90, Q3 = 91–135, Q4 = 136–181). Referência: 25% por quartil indica ausência de viés temporal.

### Métrica 2 — Taxa de detecção de eventos estruturais (TDE)
Fração dos dias de evento (pico extremo, vale/outlier, ruptura abrupta) que foram incluídos na seleção de cada método. Limiares por série:
- Pico extremo: `valor > μ + 1,5σ`
- Vale / outlier: `valor < μ − 1,5σ`
- Ruptura abrupta: `|valor[t] − valor[t−1]| > 1,5σ_diff`

### Métrica 3 — Índice de Priorização de Eventos (IPE)
- **IPE de seleção** (todos os métodos): razão entre a fração dos pontos selecionados que são eventos e a fração de dias de evento na série. IPE > 1 → prioriza eventos; IPE < 1 → subestima.
- **IPE de importância** (somente SHAP e LIME): razão entre a contribuição média absoluta nos dias de evento e nos dias sem evento, entre os pontos selecionados.

### Métrica 4 — Posição dos eventos no ranking SHAP/LIME
SHAP e LIME ranqueiam todos os 181 dias por magnitude de contribuição. Posição média dos dias de evento nesse ranking (referência: posição 91 = acaso). Para os LLMs, reporta-se a TDE como equivalente funcional.

## Resultados (conjunto de teste — 7 séries de 181 dias)

### Cobertura de quartis

| Método     | Q1 (1–45) | Q2 (46–90) | Q3 (91–135) | Q4 (136–181) |
|------------|-----------|------------|-------------|--------------|
| SHAP-RF    | 24,9%     | 24,9%      | 24,9%       | 25,4%        |
| LIME-RF    | 24,9%     | 24,9%      | 24,9%       | 25,4%        |
| LLM bruto  | 53,2%     | 12,9%      | 27,4%       | 6,5%         |
| LLM estat. | 53,8%     | 16,9%      | 10,8%       | 18,5%        |
| LLM global | 58,6%     | 20,0%      | 11,4%       | 10,0%        |

SHAP e LIME distribuem uniformemente (~25% por quartil). Os LLMs concentram 53–59% dos pontos no Q1, indicando viés para o início da série.

### Taxa de detecção de eventos (TDE)

| Método     | Picos extremos | Vales / outliers | Rupturas abruptas |
|------------|---------------|------------------|-------------------|
| SHAP-RF    | 100,0%        | 100,0%           | 100,0%            |
| LIME-RF    | 100,0%        | 100,0%           | 100,0%            |
| LLM bruto  | 9,4%          | 6,9%             | 4,8%              |
| LLM estat. | 25,5%         | 9,7%             | 8,0%              |
| LLM global | 32,1%         | 9,7%             | 10,5%             |

SHAP e LIME atingem 100% por cobrirem todos os dias. Os LLMs priorizam picos; rupturas abruptas são detectadas em menos de 11% dos casos.

### IPE de seleção

| Método     | Picos extremos | Vales / outliers | Rupturas abruptas |
|------------|---------------|------------------|-------------------|
| SHAP-RF    | 1,000         | 1,000            | 1,000             |
| LIME-RF    | 1,000         | 1,000            | 1,000             |
| LLM bruto  | 1,905         | 0,887            | 1,101             |
| LLM estat. | 5,003         | 1,430            | 1,662             |
| LLM global | 6,285         | 2,416            | 2,038             |

O LLM global prioriza picos 6× acima do acaso. O LLM bruto subestima vales (IPE < 1).

### IPE de importância — SHAP e LIME

| Evento             | SHAP-RF | LIME-RF |
|--------------------|---------|---------|
| Picos extremos     | 0,837   | 0,833   |
| Vales / outliers   | 1,515   | 1,512   |
| Rupturas abruptas  | 0,990   | 0,979   |

O Random Forest atribui **menor** contribuição absoluta aos picos do que aos dias comuns (IPE < 1), e **maior** contribuição às depressões (IPE ≈ 1,51). O modelo aprendeu a discriminar perfis pelos vales, não pelos picos.

### Posição dos eventos no ranking SHAP/LIME (referência: 91 = acaso)

| Evento             | SHAP-RF | LIME-RF |
|--------------------|---------|---------|
| Picos extremos     | 100,9   | 94,9    |
| Vales / outliers   | 89,2    | 89,8    |
| Rupturas abruptas  | 93,4    | 94,0    |

Picos extremos ficam **abaixo do acaso** no ranking SHAP (posição 100,9 > 91), confirmando que o RF os trata como menos discriminativos. Vales ficam levemente acima do acaso.

## Interface Streamlit

| Aba              | Conteúdo                                                                 |
|------------------|--------------------------------------------------------------------------|
| Base             | Tabela da base sintética e visualização de cada série                    |
| Métricas         | Acurácia, F1, precisão e recall do Random Forest                         |
| Resultados       | Tabela de resultados do pipeline por série                               |
| Detalhe por série | Gráfico interativo com pontos SHAP, LIME e LLM sobrepostos à série      |
| Relatório        | Painéis de 5 subgráficos (um por método) para Perfil A e Perfil B       |
| Métricas XAI     | Heatmaps das 4 métricas comparativas                                     |

### Controles da aba Relatório
- **% dos pontos SHAP e LIME mais significativos** — slider para filtrar os top-N% por magnitude.
- **Visualizar importância** — quando ativo, linhas ficam cinza e os X são coloridos por nível: verde escuro (top 33%), laranja (médio 33%), vermelho (baixo 33%).

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Configuração da API

Crie `.env` na raiz do projeto:

```env
OPENAI_API_KEY=sua_chave_aqui
```

O modelo padrão é `gpt-4.1-mini` (configurável em `config.py`).

## Execução

```bash
streamlit run app.py
```

Na barra lateral:
1. Clique em **Gerar base sintética**.
2. Clique em **Executar pipeline**.
3. Navegue pelas abas para explorar os resultados.

## Arquivos principais

| Arquivo                   | Descrição                                                      |
|---------------------------|----------------------------------------------------------------|
| `app.py`                  | Interface Streamlit com todas as abas                          |
| `executar_pipeline.py`    | Orquestra treino, SHAP, LIME e chamadas ao LLM                 |
| `treinar_modelos.py`      | Treino e avaliação do Random Forest                            |
| `explicabilidade_pontos.py` | Cálculo de pontos SHAP e LIME por dia                        |
| `teste_llm_modelos.py`    | Construção do prompt e chamada à API OpenAI                    |
| `gerar_base.py`           | Geração da base sintética com motivos de especialista          |
| `features_series.py`      | Extração de colunas `dia_*` da série temporal                  |
| `gerar_relatorio_pdf.py`  | Geração de relatório PDF com gráficos e tabelas das métricas   |
| `utils_json.py`           | Extração de JSON da resposta do LLM                            |
| `config.py`               | Caminhos e constantes do projeto                               |

## Saídas

| Arquivo                                         | Conteúdo                                        |
|-------------------------------------------------|-------------------------------------------------|
| `data/series_temporais_educacionais.csv`        | Base sintética completa                         |
| `outputs/resultados/comparacao_resultados.csv`  | Resultados por série: pontos SHAP, LIME e LLM  |
| `outputs/resultados/modelo_random_forest.joblib`| Modelo treinado + métricas                      |
| `outputs/resultados/resultado_llm_resumo.csv`   | Respostas brutas do LLM por série               |
| `outputs/metodologia_e_resultados.txt`          | Metodologia das métricas e resultados numéricos |
