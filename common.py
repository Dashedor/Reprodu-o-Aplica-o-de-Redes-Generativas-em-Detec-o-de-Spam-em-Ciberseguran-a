"""Funções compartilhadas. Todo CSV de entrada precisa ter as colunas CONTENT (texto) e CLASS (1=Spam, 0=Ham)."""
from pathlib import Path
import re

import pandas as pd

SAIDA = Path("saida")
SAIDA.mkdir(exist_ok=True)


def limpar(texto) -> str:
    texto = str(texto).replace("\ufeff", "")          # BOM presente no fim dos comentários
    return re.sub(r"\s+", " ", texto).strip()


def carregar_csv(caminho) -> pd.DataFrame:
    df = pd.read_csv(caminho)
    faltando = {"CONTENT", "CLASS"} - set(df.columns)
    if faltando:
        raise ValueError(f"{caminho}: faltam as colunas {faltando}")
    df["CONTENT"] = df["CONTENT"].map(limpar)
    df = df[df["CONTENT"].str.len() > 0].reset_index(drop=True)
    df["CLASS"] = df["CLASS"].astype(int)
    return df


def nome_do_arquivo(caminho) -> str:
    return Path(caminho).stem


def salvar_resultados(df: pd.DataFrame, caminho_csv, index, columns) -> None:
    """Grava dois CSVs: o resultado completo (formato longo) e uma tabela pivotada no estilo do artigo
    (<nome>_tabela.csv), com colunas do tipo 'BoW_Acc', 'TF-IDF_ROC_AUC', 'BERT_Entropia-Cruzada_Acc'..."""
    caminho_csv = Path(caminho_csv)
    df.to_csv(caminho_csv, index=False)
    tab = df.pivot_table(index=index, columns=columns, values=["Acc", "Precision", "ROC_AUC"]).round(4)
    # nível das métricas passa a ser o último no nome da coluna
    tab.columns = ["_".join(str(x) for x in (c[1:] + (c[0],))) for c in tab.columns.to_flat_index()]
    tab = tab.reset_index()
    destino = caminho_csv.with_name(caminho_csv.stem + "_tabela.csv")
    tab.to_csv(destino, index=False)
    print(f"Resultados salvos em:\n  {caminho_csv}  (formato longo)\n  {destino}  (tabela comparativa)")
