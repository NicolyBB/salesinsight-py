# SalesInsight PY

Projeto de análise de vendas usando Python.

O objetivo é pegar uma base de vendas, fazer a limpeza dos dados e gerar informações que ajudem a entender melhor os resultados.

## O que o projeto faz

- lê o arquivo `vendas.csv`;
- mostra uma visão inicial dos dados;
- limpa datas, textos, quantidades e preços;
- remove registros que não podem ser usados na análise;
- calcula a receita de cada venda;
- cria mês, ano, trimestre e faixa de receita;
- calcula métricas por mês, produto, categoria e região;
- classifica clientes em Bronze, Prata e Ouro;
- faz cálculos com NumPy;
- gera quatro gráficos;
- salva os resultados em CSV e JSON.

## Bibliotecas usadas

- Pandas
- NumPy
- Matplotlib
- Seaborn
- JSON
- OS
- RE

## Conceitos usados

Durante o projeto foram usados:

- variáveis e tipos de dados;
- operadores;
- `if`, `elif` e `else`;
- repetição com `for`;
- listas e dicionários;
- funções com parâmetros, retorno e docstrings;
- funções `lambda`;
- função que recebe outra função como argumento;
- leitura e escrita de CSV e JSON;
- datas com Pandas;
- expressões regulares com `re`;
- DataFrames, filtros e `groupby`;
- arrays NumPy, operações vetorizadas e broadcasting;
- `np.select`;
- gráficos com Matplotlib e Seaborn;
- classe `AnalisadorDeVendas`.

## Como executar no Google Colab

1. Abra um notebook novo no Google Colab.
2. Envie `salesinsight.py` e `vendas.csv` pela pasta lateral.
3. Execute:

```python
!python salesinsight.py
```

## Como executar no computador

Instale as bibliotecas:

```bash
pip install -r requirements.txt
```

Depois execute:

```bash
python salesinsight.py
```

## Arquivos gerados

Na pasta `outputs`:

- `dados_limpos.csv`
- `metricas_por_mes.csv`
- `segmentacao_clientes.csv`
- `estatisticas_gerais.json`

Na pasta `outputs/graficos`:

- `receita_por_mes.png`
- `top_produtos.png`
- `quantidade_vs_receita.png`
- `painel_resumo.png`

## Organização do projeto

O Kanban está no arquivo:

`planejamento/tarefas-kanban.md`

## Vídeo de demonstração

**Link do vídeo:** 

