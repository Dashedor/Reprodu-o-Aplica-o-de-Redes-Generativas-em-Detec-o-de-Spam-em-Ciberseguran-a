"""(Opcional) Divide o dataset original em treino/teste (70/30, estratificado).

python 00_dividir_dataset.py --csv TWT_Dataset.csv      -> saida/treino.csv e saida/teste.csv
"""
import argparse

from sklearn.model_selection import train_test_split

from common import SAIDA, carregar_csv

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="TWT_Dataset.csv")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--teste", type=float, default=0.3)
a = ap.parse_args()

df = carregar_csv(a.csv)
tr, te = train_test_split(df, test_size=a.teste, stratify=df["CLASS"], random_state=a.seed)
tr.to_csv(SAIDA / "treino.csv", index=False)
te.to_csv(SAIDA / "teste.csv", index=False)
print(f"treino={len(tr)}  teste={len(te)}  -> {SAIDA}/treino.csv, {SAIDA}/Benigno.csv")
