# Projeto RF + LLM — versão essencial

Objetivo: classificar séries temporais sintéticas com Random Forest e verificar se o LLM explica a classificação usando motivos compatíveis com os motivos da especialista.

## Fluxo

1. Geração de base sintética com `perfil_A` e `perfil_B`.
2. Treino do Random Forest usando apenas colunas `dia_*`.
3. Classificação da base de teste.
4. Envio ao LLM de:
   - série temporal testada;
   - classificação do Random Forest;
   - probabilidades do Random Forest;
   - hiperparâmetros do Random Forest.
5. O LLM não recebe:
   - exemplos classificados pelo RF;
   - motivos da especialista;
   - explicação ground truth.
6. O sistema compara os motivos indicados pelo LLM com os motivos reais da especialista.

## Instalação

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Configuração da API

Crie um arquivo `.env` na raiz do projeto:

```env
OPENAI_API_KEY=sua_chave_aqui
```

## Execução

```bash
python -m streamlit run app.py
```

## Arquivos principais

- `gerar_base.py`: gera a base sintética e os motivos da especialista.
- `treinar_random_forest.py`: treina o Random Forest.
- `teste_llm_resumo_rf.py`: monta o prompt e chama o LLM.
- `executar_pipeline.py`: executa o experimento e compara LLM × especialista.
- `app.py`: interface Streamlit.

## Visualização SHAP/LIME por ponto da série

Esta versão acrescenta uma camada visual de explicabilidade local na aba **Detalhe por série**.

### O que foi acrescentado

- Cálculo de contribuição por ponto/dia da série usando SHAP.
- Cálculo de contribuição por ponto/dia da série usando LIME.
- Gráfico interativo com seleção entre:
  - `SHAP`
  - `LIME`
  - `SHAP + LIME`
- Controle para escolher quantos pontos mais relevantes serão destacados.
- Tabela com ranking, dia, data, valor, contribuição e sentido da contribuição.

### Como interpretar o gráfico

- A linha representa a série temporal original.
- A linha tracejada representa a média da série.
- Os marcadores indicam os pontos destacados pelo método escolhido.
- Marcadores maiores indicam maior contribuição absoluta.
- Contribuição positiva favorece a classe prevista pelo Random Forest.
- Contribuição negativa reduz a força da classe prevista pelo Random Forest.

### Execução atualizada

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute o Streamlit:

```bash
python -m streamlit run app.py
```

Na barra lateral:

1. Clique em **Gerar base sintética**.
2. Marque **Executar SHAP e LIME**.
3. Clique em **Executar pipeline**.
4. Abra a aba **Detalhe por série**.
5. Escolha `SHAP`, `LIME` ou `SHAP + LIME`.

### Arquivo novo

- `explicabilidade_pontos.py`: concentra o cálculo e a organização dos pontos explicados por SHAP e LIME.

### Colunas novas em `outputs/resultados/comparacao_resultados.csv`

- `shap_pontos`
- `lime_pontos`
- `shap_top_10`
- `lime_top_10`
- `erro_xai`


## Atualização: comparação SHAP × LIME × LLM por ponto

Esta versão adiciona uma camada de explicabilidade declarada pelo LLM. O LLM recebe somente a série temporal de teste e retorna:

- `classe_interpretada_pelo_llm`
- `motivos_identificados_pelo_llm`
- `explicacao_do_llm`
- `llm_pontos`: pontos que o LLM considerou relevantes, com dia, valor, importância, motivo associado e justificativa

Na interface Streamlit, a aba **Detalhe por série** permite visualizar:

- SHAP
- LIME
- LLM
- SHAP + LIME
- SHAP + LIME + LLM

Também foram adicionadas colunas de comparação entre os dias destacados pelos métodos:

- `dias_comuns_shap_lime`
- `dias_comuns_shap_llm`
- `dias_comuns_lime_llm`
- `dias_comuns_todos`

Para usar a camada LLM, configure `OPENAI_API_KEY` no `.env`, marque **Executar LLM** na barra lateral e execute o pipeline novamente.
