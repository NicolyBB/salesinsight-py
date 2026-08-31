"""SalesInsight PY

Projeto de analise de vendas usando Python.

- ler os dados;
- limpar o que estiver errado;
- criar algumas informacoes novas;
- fazer calculos;
- gerar graficos;
- salvar os resultados.
"""

import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PASTA_ATUAL = os.path.dirname(os.path.abspath(__file__))
PASTA_OUTPUTS = os.path.join(PASTA_ATUAL, "outputs")
PASTA_GRAFICOS = os.path.join(PASTA_OUTPUTS, "graficos")


def carregar_dados(caminho):
    """Le o arquivo CSV."""
    return pd.read_csv(caminho)


def mostrar_inicio(df):
    """Mostra algumas informacoes para conhecer os dados."""
    print("\n=== DADOS INICIAIS ===")
    print("Tamanho:", df.shape)
    print("\nColunas:", list(df.columns))
    print("\nTipos:\n", df.dtypes)
    print("\nValores vazios:\n", df.isnull().sum())
    print("\nPrimeiras linhas:\n", df.head())


def arrumar_preco(valor):
    """Transforma o preco em numero."""
    if pd.isna(valor):
        return np.nan

    if isinstance(valor, (int, float, np.integer, np.floating)):
        return float(valor)

    texto = str(valor).strip()
    texto = re.sub(r"[^0-9,.-]", "", texto)

    if texto == "":
        return np.nan

    if "," in texto:
        texto = texto.replace(".", "").replace(",", ".")

    try:
        return float(texto)
    except ValueError:
        return np.nan


def arrumar_quantidade(valor):
    """Pega o numero da quantidade, inclusive em textos como '6 un'."""
    if pd.isna(valor):
        return np.nan

    numero = re.search(r"-?\d+", str(valor))

    if numero:
        return float(numero.group())

    return np.nan


def arrumar_datas(coluna):
    """Tenta ler os formatos de data encontrados na base."""
    resultado = pd.Series(pd.NaT, index=coluna.index, dtype="datetime64[ns]")
    formatos = ["%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"]

    for formato in formatos:
        faltando = resultado.isna()

        if not faltando.any():
            break

        resultado.loc[faltando] = pd.to_datetime(
            coluna.loc[faltando].astype(str).str.strip(),
            format=formato,
            errors="coerce"
        )

    return resultado


def limpar_dados(df):
    """Limpa os dados que serao usados na analise."""
    df = df.copy()
    quantidade_inicial = len(df)

    # Limpeza dos textos
    for coluna in ["cliente", "produto", "categoria", "regiao"]:
        df[coluna] = df[coluna].fillna("").astype(str).str.strip()

    df["produto"] = df["produto"].str.replace(r"\s+", " ", regex=True)
    df["categoria"] = df["categoria"].str.replace(r"\s+", " ", regex=True).str.title()
    df["regiao"] = df["regiao"].str.replace(r"\s+", " ", regex=True).str.title()

    # Datas
    df["data_venda"] = arrumar_datas(df["data_venda"])
    datas_invalidas = int(df["data_venda"].isna().sum())
    df = df.dropna(subset=["data_venda"]).copy()

    # Quantidade e preco
    df["quantidade"] = df["quantidade"].apply(arrumar_quantidade)
    df["preco_unitario"] = df["preco_unitario"].apply(arrumar_preco)

    nulos_numericos = int(
        df[["quantidade", "preco_unitario"]].isna().any(axis=1).sum()
    )

    df = df.dropna(subset=["quantidade", "preco_unitario"]).copy()

    # Valores iguais ou menores que zero nao entram na analise
    valores_invalidos = int(
        ((df["quantidade"] <= 0) | (df["preco_unitario"] <= 0)).sum()
    )

    df = df[
        (df["quantidade"] > 0) &
        (df["preco_unitario"] > 0)
    ].copy()

    df["quantidade"] = df["quantidade"].astype(int)
    df["preco_unitario"] = df["preco_unitario"].astype(float)

    # Padronizacao dos clientes
    padrao_cliente = re.compile(r"^Cliente_\d{3}$", flags=re.IGNORECASE)

    def limpar_nome(nome):
        texto = re.sub(r"[^A-Za-zÀ-ÿ0-9 ]", " ", str(nome).strip())
        texto = re.sub(r"\s+", " ", texto).strip()
        return texto.title()

    df["cliente_nome"] = df["cliente"].apply(limpar_nome)

    clientes_vazios = int((df["cliente_nome"] == "").sum())
    df = df[df["cliente_nome"] != ""].copy()

    fora_padrao_antes = int(
        (~df["cliente"].astype(str).str.match(padrao_cliente, na=False)).sum()
    )

    nomes = sorted(df["cliente_nome"].unique())
    mapa = {}

    for i, nome in enumerate(nomes, start=1):
        mapa[nome] = f"Cliente_{i:03d}"

    df["cliente"] = df["cliente_nome"].map(mapa)

    fora_padrao_depois = int(
        (~df["cliente"].str.match(padrao_cliente)).sum()
    )

    quantidade_final = len(df)

    relatorio = {
        "registros_iniciais": int(quantidade_inicial),
        "datas_invalidas_removidas": datas_invalidas,
        "nulos_numericos_removidos": nulos_numericos,
        "valores_nao_positivos_removidos": valores_invalidos,
        "clientes_vazios_removidos": clientes_vazios,
        "clientes_fora_padrao_antes": fora_padrao_antes,
        "clientes_fora_padrao_depois": fora_padrao_depois,
        "registros_removidos_total": int(quantidade_inicial - quantidade_final),
        "registros_finais": int(quantidade_final)
    }

    print("\n=== LIMPEZA ===")
    for nome, valor in relatorio.items():
        print(nome, ":", valor)

    return df.reset_index(drop=True), relatorio


def criar_colunas(df):
    """Cria colunas novas para ajudar na analise."""
    df = df.copy()

    df["receita_total"] = df["quantidade"] * df["preco_unitario"]
    df["mes"] = df["data_venda"].dt.month
    df["ano"] = df["data_venda"].dt.year
    df["trimestre"] = "Q" + df["data_venda"].dt.quarter.astype(str)

    meses = {
        1: "Janeiro",
        2: "Fevereiro",
        3: "Marco",
        4: "Abril",
        5: "Maio",
        6: "Junho",
        7: "Julho",
        8: "Agosto",
        9: "Setembro",
        10: "Outubro",
        11: "Novembro",
        12: "Dezembro"
    }

    df["mes_nome"] = df["mes"].map(meses)

    condicoes = [
        df["receita_total"] < 500,
        (df["receita_total"] >= 500) & (df["receita_total"] < 5000),
        df["receita_total"] >= 5000
    ]

    faixas = ["Baixo Valor", "Medio Valor", "Alto Valor"]

    df["faixa_receita_item"] = np.select(
        condicoes,
        faixas,
        default="Nao Classificado"
    )

    return df


def processar_coluna(df, coluna, funcao, nome_nova_coluna=None):
    """Recebe uma funcao e aplica em uma coluna."""
    df = df.copy()

    if nome_nova_coluna is None:
        nome_nova_coluna = coluna + "_transformado"

    df[nome_nova_coluna] = df[coluna].apply(funcao)
    return df


def calcular_metricas(df):
    """Faz os agrupamentos principais."""
    por_mes = (
        df.groupby(["ano", "mes", "mes_nome"], as_index=False)
        .agg(
            receita_total=("receita_total", "sum"),
            quantidade=("quantidade", "sum"),
            n_vendas=("id_venda", "count")
        )
        .sort_values(["ano", "mes"])
    )

    top_produtos = (
        df.groupby("produto", as_index=False)["receita_total"]
        .sum()
        .sort_values("receita_total", ascending=False)
        .head(5)
    )

    por_categoria = (
        df.groupby("categoria", as_index=False)["receita_total"]
        .sum()
        .sort_values("receita_total", ascending=False)
    )

    por_regiao = (
        df.groupby("regiao", as_index=False)
        .agg(
            receita_total=("receita_total", "sum"),
            ticket_medio=("receita_total", "mean")
        )
        .sort_values("receita_total", ascending=False)
    )

    metricas = {
        "por_mes": por_mes,
        "top_produtos": top_produtos,
        "por_categoria": por_categoria,
        "por_regiao": por_regiao
    }

    print("\n=== METRICAS ===")

    for nome, tabela in metricas.items():
        print("\n", nome.upper())
        print(tabela.to_string(index=False))

    return metricas


def segmentar_clientes(df):
    """Classifica os clientes em Bronze, Prata e Ouro."""
    clientes = (
        df.groupby(["cliente", "cliente_nome"], as_index=False)["receita_total"]
        .sum()
        .rename(columns={"receita_total": "total_gasto"})
    )

    def escolher_segmento(gasto):
        if gasto < 5000:
            return "Bronze"
        elif gasto <= 15000:
            return "Prata"
        else:
            return "Ouro"

    clientes["segmento"] = clientes["total_gasto"].apply(
        lambda valor: escolher_segmento(valor)
    )

    clientes = clientes.sort_values(
        "total_gasto",
        ascending=False
    ).reset_index(drop=True)

    print("\n=== TOP 10 CLIENTES ===")
    print(clientes.head(10).to_string(index=False))

    print("\n=== SEGMENTOS ===")
    print(clientes["segmento"].value_counts())

    return clientes


def calcular_numpy(df):
    """Faz calculos simples com NumPy."""
    receitas = df["receita_total"].to_numpy(dtype=float)

    media = np.mean(receitas)
    mediana = np.median(receitas)
    desvio = np.std(receitas)
    soma = np.sum(receitas)
    minimo = np.min(receitas)
    maximo = np.max(receitas)

    # Escala de 0 a 1 usando broadcasting
    amplitude = maximo - minimo

    if amplitude != 0:
        escala = (receitas - minimo) / amplitude
    else:
        escala = np.zeros_like(receitas)

    # Vendas acima da media
    acima_da_media = receitas[receitas > media]

    estatisticas = {
        "media": float(media),
        "mediana": float(mediana),
        "desvio_padrao_ddof_0": float(desvio),
        "soma": float(soma),
        "minimo": float(minimo),
        "maximo": float(maximo),
        "vendas_acima_da_media": int(acima_da_media.size),
        "escala_0_1_min": float(np.min(escala)),
        "escala_0_1_max": float(np.max(escala))
    }

    print("\n=== NUMPY ===")
    for nome, valor in estatisticas.items():
        print(nome, ":", round(valor, 2) if isinstance(valor, float) else valor)

    return estatisticas


def criar_graficos(df, metricas):
    """Cria os quatro graficos do projeto."""
    os.makedirs(PASTA_GRAFICOS, exist_ok=True)
    sns.set_theme(style="whitegrid")

    por_mes = metricas["por_mes"].copy()
    top = metricas["top_produtos"].copy()
    por_regiao = metricas["por_regiao"].copy()

    por_mes["periodo"] = (
        por_mes["ano"].astype(str)
        + "-"
        + por_mes["mes"].astype(str).str.zfill(2)
    )

    # 1 - Receita por mes
    plt.figure(figsize=(10, 5))
    plt.plot(
        por_mes["periodo"],
        por_mes["receita_total"],
        marker="o"
    )
    plt.title("Receita por Mes")
    plt.xlabel("Periodo")
    plt.ylabel("Receita (R$)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PASTA_GRAFICOS, "receita_por_mes.png"),
        dpi=150
    )
    plt.close()

    # 2 - Produtos
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=top,
        y="produto",
        x="receita_total",
        hue="produto",
        legend=False,
        palette="pastel"
    )
    plt.title("Top 5 Produtos por Receita")
    plt.xlabel("Receita (R$)")
    plt.ylabel("Produto")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PASTA_GRAFICOS, "top_produtos.png"),
        dpi=150
    )
    plt.close()

    # 3 - Quantidade e receita
    plt.figure(figsize=(10, 5))
    sns.scatterplot(
        data=df,
        x="quantidade",
        y="receita_total",
        hue="categoria",
        palette="Set2"
    )
    plt.title("Quantidade x Receita")
    plt.xlabel("Quantidade")
    plt.ylabel("Receita (R$)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PASTA_GRAFICOS, "quantidade_vs_receita.png"),
        dpi=150
    )
    plt.close()

    # 4 - Painel 2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(
        por_mes["periodo"],
        por_mes["receita_total"],
        marker="o"
    )
    axes[0, 0].set_title("Receita por Mes")
    axes[0, 0].set_xlabel("Periodo")
    axes[0, 0].set_ylabel("Receita (R$)")
    axes[0, 0].tick_params(axis="x", rotation=45)

    axes[0, 1].barh(
        top["produto"],
        top["receita_total"]
    )
    axes[0, 1].set_title("Top Produtos")
    axes[0, 1].set_xlabel("Receita (R$)")
    axes[0, 1].set_ylabel("Produto")

    sns.scatterplot(
        data=df,
        x="quantidade",
        y="receita_total",
        hue="categoria",
        palette="Set2",
        ax=axes[1, 0]
    )
    axes[1, 0].set_title("Quantidade x Receita")
    axes[1, 0].set_xlabel("Quantidade")
    axes[1, 0].set_ylabel("Receita (R$)")

    axes[1, 1].bar(
        por_regiao["regiao"],
        por_regiao["receita_total"]
    )
    axes[1, 1].set_title("Receita por Regiao")
    axes[1, 1].set_xlabel("Regiao")
    axes[1, 1].set_ylabel("Receita (R$)")
    axes[1, 1].tick_params(axis="x", rotation=30)

    fig.suptitle("Painel Resumo")
    plt.tight_layout()

    plt.savefig(
        os.path.join(PASTA_GRAFICOS, "painel_resumo.png"),
        dpi=150
    )
    plt.close()

    print("\nGraficos criados em:", PASTA_GRAFICOS)


def salvar_resultados(df, metricas, clientes, estatisticas, relatorio):
    """Salva os arquivos criados pelo projeto."""
    os.makedirs(PASTA_OUTPUTS, exist_ok=True)

    df.to_csv(
        os.path.join(PASTA_OUTPUTS, "dados_limpos.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    metricas["por_mes"].to_csv(
        os.path.join(PASTA_OUTPUTS, "metricas_por_mes.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    clientes.to_csv(
        os.path.join(PASTA_OUTPUTS, "segmentacao_clientes.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    conteudo = {
        "estatisticas_numpy": estatisticas,
        "relatorio_limpeza": relatorio
    }

    caminho_json = os.path.join(
        PASTA_OUTPUTS,
        "estatisticas_gerais.json"
    )

    with open(caminho_json, "w", encoding="utf-8") as arquivo:
        json.dump(
            conteudo,
            arquivo,
            indent=4,
            ensure_ascii=False
        )

    # Le o JSON novamente para conferir
    with open(caminho_json, "r", encoding="utf-8") as arquivo:
        leitura = json.load(arquivo)

    print("\n=== JSON SALVO ===")
    print(json.dumps(leitura, indent=2, ensure_ascii=False))


class AnalisadorDeVendas:
    """Organiza as etapas do projeto."""

    def __init__(self, caminho_arquivo):
        self.caminho_arquivo = caminho_arquivo
        self.df_bruto = None
        self.df_limpo = None
        self.metricas = {}
        self.clientes = None
        self.estatisticas = {}
        self.relatorio = {}

    def carregar(self):
        self.df_bruto = carregar_dados(self.caminho_arquivo)

    def limpar(self):
        self.df_limpo, self.relatorio = limpar_dados(self.df_bruto)

    def transformar(self):
        self.df_limpo = criar_colunas(self.df_limpo)

        # Receita em milhares
        self.df_limpo = processar_coluna(
            self.df_limpo,
            "receita_total",
            lambda valor: round(valor / 1000, 2),
            "receita_em_milhares"
        )

        # Classificacao do volume
        self.df_limpo = processar_coluna(
            self.df_limpo,
            "quantidade",
            lambda valor: "Alto Volume" if valor > 5 else "Baixo Volume",
            "perfil_volume"
        )

    def analisar(self):
        self.metricas = calcular_metricas(self.df_limpo)
        self.clientes = segmentar_clientes(self.df_limpo)
        self.estatisticas = calcular_numpy(self.df_limpo)

    def visualizar(self):
        criar_graficos(self.df_limpo, self.metricas)

    def exportar(self):
        salvar_resultados(
            self.df_limpo,
            self.metricas,
            self.clientes,
            self.estatisticas,
            self.relatorio
        )

    def mostrar_resumo(self):
        total = self.df_limpo["receita_total"].sum()
        quantidade = self.df_limpo["quantidade"].sum()

        top_produto = self.metricas["top_produtos"].iloc[0]
        top_regiao = self.metricas["por_regiao"].iloc[0]

        print("\n=== RESUMO FINAL ===")
        print("Registros validos:", len(self.df_limpo))
        print("Receita total: R$", round(total, 2))
        print("Quantidade vendida:", quantidade)
        print(
            "Produto com maior receita:",
            top_produto["produto"],
            "- R$",
            round(top_produto["receita_total"], 2)
        )
        print(
            "Regiao com maior receita:",
            top_regiao["regiao"],
            "- R$",
            round(top_regiao["receita_total"], 2)
        )


def main():
    """Executa o projeto."""
    print("=" * 50)
    print("SALESINSIGHT PY")
    print("=" * 50)

    caminho = os.path.join(PASTA_ATUAL, "vendas.csv")

    if not os.path.exists(caminho):
        print("Erro: coloque o arquivo vendas.csv na mesma pasta.")
        return

    analisador = AnalisadorDeVendas(caminho)

    analisador.carregar()
    mostrar_inicio(analisador.df_bruto)

    analisador.limpar()
    analisador.transformar()
    analisador.analisar()
    analisador.visualizar()
    analisador.exportar()
    analisador.mostrar_resumo()

    print("\nProjeto executado com sucesso.")


if __name__ == "__main__":
    main()
