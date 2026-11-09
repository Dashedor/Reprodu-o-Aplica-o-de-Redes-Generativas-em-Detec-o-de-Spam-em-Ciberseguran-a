"""BERT e RoBERTa (PyTorch + Hugging Face) com Entropia Cruzada ou Dual Contrastive Learning.

pip install torch transformers scikit-learn pandas
Você informa o(s) CSV(s) de treino e o(s) CSV(s) de teste (benignos ou envenenados).

Padrão (hiperparâmetros fixos, como no artigo):
  python 03_modelos_deep_learning.py --treino saida/treino.csv saida/treino_gpt4o.csv --teste saida/teste.csv

Com otimização (--otimizar): busca aleatória de hiperparâmetros (lr, épocas, batch, max_len e tau na perda Dual)
para cada (modelo, perda), avaliando num conjunto de VALIDAÇÃO separado do treino benigno (--base; padrão =
1º --treino). O teste nunca é usado na escolha. A config padrão é sempre uma das tentativas. Os melhores
parâmetros são salvos em CSV e reutilizados em todos os treinos (benigno e envenenados).
  python 03_modelos_deep_learning.py --otimizar --n-tentativas 8 --treino saida/treino.csv saida/treino_gpt4o.csv --teste saida/teste.csv

Para reaproveitar parâmetros já encontrados:  --params saida/melhores_parametros_deep.csv
"""
import argparse
import gc
import itertools
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from common import SAIDA, carregar_csv, nome_do_arquivo, salvar_resultados

CHECKPOINTS = {"BERT": "bert-base-uncased", "RoBERTa": "roberta-base"}


def fixar_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class Classificador(nn.Module):
    """Encoder + camada linear. Os pesos da camada linear (θ) funcionam como representações das classes."""

    def __init__(self, checkpoint, perda, n_classes=2, dropout=0.1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(checkpoint)
        self.drop = nn.Dropout(dropout)
        self.perda = perda
        self.fc = nn.Linear(self.enc.config.hidden_size, n_classes, bias=(perda == "ce"))

    def forward(self, ids, mask):
        z = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]   # token [CLS]/<s>
        z = self.drop(z)
        return z, self.fc(z)


def supcon(z, y, tau):
    """L_z: contraste supervisionado entre representações z (mesmo rótulo = par positivo)."""
    z = F.normalize(z, dim=1)
    sim = (z @ z.T) / tau
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (y[:, None] == y[None, :]) & ~eye
    npos = pos.sum(1)
    ok = npos > 0
    if not ok.any():
        return z.sum() * 0.0
    return (-(logp * pos).sum(1)[ok] / npos[ok]).mean()


def logits_cos(z, fc, tau):
    return F.normalize(z, dim=1) @ F.normalize(fc.weight, dim=1).T / tau


def calcular_perda(modelo, z, logits, y, tau):
    if modelo.perda == "ce":
        return F.cross_entropy(logits, y)
    # Dual: L = L_z + L_θ  (L_θ = contraste entre z e os vetores-classe θ)
    l_z = supcon(z, y, tau)
    l_theta = F.cross_entropy(logits_cos(z, modelo.fc, tau), y)
    return l_z + l_theta


def tokenizar(tok, textos, max_len):
    enc = tok(list(textos), padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


@torch.no_grad()
def avaliar(modelo, loader, dev, tau):
    modelo.eval()
    probs, ys = [], []
    for ids, mask, y in loader:
        z, logits = modelo(ids.to(dev), mask.to(dev))
        if modelo.perda == "dual":
            logits = logits_cos(z, modelo.fc, tau)
        probs.append(F.softmax(logits, dim=1)[:, 1].cpu()); ys.append(y)
    p, y = torch.cat(probs).numpy(), torch.cat(ys).numpy()
    pred = (p >= 0.5).astype(int)
    return {"Acc": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0), "F1": f1_score(y, pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y, p)}


def rodar(nome_modelo, perda, treino, testes, args, dev):
    """Treina uma vez e avalia em todos os testes. Retorna {nome_teste: métricas}."""
    fixar_seed(args.seed)
    ck = CHECKPOINTS[nome_modelo]
    tok = AutoTokenizer.from_pretrained(ck)
    ids_tr, m_tr = tokenizar(tok, treino["CONTENT"], args.max_len)
    y_tr = torch.tensor(treino["CLASS"].values)
    dl_tr = DataLoader(TensorDataset(ids_tr, m_tr, y_tr), batch_size=args.batch, shuffle=True)

    modelo = Classificador(ck, perda).to(dev)
    opt = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(dl_tr) * args.epocas
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    for ep in range(args.epocas):
        modelo.train()
        soma = 0.0
        for ids, mask, y in dl_tr:
            ids, mask, y = ids.to(dev), mask.to(dev), y.to(dev)
            z, logits = modelo(ids, mask)
            loss = calcular_perda(modelo, z, logits, y, args.tau)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step(); sched.step()
            soma += loss.item()
        print(f"    época {ep + 1}/{args.epocas} loss={soma / len(dl_tr):.4f}")

    res = {}
    for nome_te, te in testes.items():
        ids_te, m_te = tokenizar(tok, te["CONTENT"], args.max_len)
        dl_te = DataLoader(TensorDataset(ids_te, m_te, torch.tensor(te["CLASS"].values)), batch_size=args.batch * 2)
        res[nome_te] = avaliar(modelo, dl_te, dev, args.tau)
    del modelo, opt; gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return res


ESPACO = {"lr": [1e-5, 2e-5, 3e-5, 5e-5], "epocas": [3, 5, 8], "batch": [8, 16, 32],
          "max_len": [64, 128], "tau": [0.05, 0.1, 0.3]}          # tau só vale para a perda Dual
METRICAS = {"f1": "F1", "roc_auc": "ROC_AUC", "accuracy": "Acc", "precision": "Precision", "recall": "Recall"}
NOME_PERDA = {"ce": "Entropia-Cruzada", "dual": "Dual Contrastive"}


def amostrar_configs(perda, padrao, n, seed):
    """Config padrão + (n-1) configs sorteadas do espaço de busca (busca aleatória, sem repetição)."""
    chaves = [k for k in ESPACO if not (k == "tau" and perda == "ce")]
    configs = [{k: padrao[k] for k in chaves}]
    todas = list(itertools.product(*(ESPACO[k] for k in chaves)))
    random.Random(seed).shuffle(todas)
    for combo in todas:
        cfg = dict(zip(chaves, combo))
        if cfg not in configs:
            configs.append(cfg)
        if len(configs) >= n:
            break
    return configs


def buscar_hiperparametros(nome, perda, base, args, dev):
    """Treina cada config em 80% do treino benigno e avalia nos 20% restantes (validação)."""
    tr, val = train_test_split(base, test_size=args.val_frac, stratify=base["CLASS"], random_state=args.seed)
    padrao = {k: getattr(args, k) for k in ESPACO}
    metrica = METRICAS[args.scoring]
    melhor, melhor_score, historico = None, -1.0, []
    configs = amostrar_configs(perda, padrao, args.n_tentativas, args.seed)
    for i, cfg in enumerate(configs, 1):
        print(f"  [busca {nome}|{perda}] tentativa {i}/{len(configs)}: {cfg}")
        hp = argparse.Namespace(**{**vars(args), **cfg})
        r = rodar(nome, perda, tr, {"val": val}, hp, dev)["val"]
        print(f"    validação: {metrica}={r[metrica]:.4f}")
        historico.append({"Modelo": nome, "Perda": perda, "Tentativa": i, "Parametros": json.dumps(cfg), **r})
        if r[metrica] > melhor_score:
            melhor, melhor_score = cfg, r[metrica]
    return melhor, melhor_score, historico


def carregar_params(caminho):
    df = pd.read_csv(caminho)
    return {(r["Modelo"], r["Perda"]): json.loads(r["Parametros"]) for _, r in df.iterrows()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treino", nargs="+", required=True, help="um ou mais CSVs de treino")
    ap.add_argument("--teste", nargs="+", required=True, help="um ou mais CSVs de teste")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--modelos", nargs="+", default=["BERT", "RoBERTa"], choices=list(CHECKPOINTS))
    ap.add_argument("--perdas", nargs="+", default=["ce", "dual"], choices=["ce", "dual"])
    ap.add_argument("--epocas", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--tau", type=float, default=0.1, help="temperatura da perda contrastiva")
    ap.add_argument("--otimizar", action="store_true", help="busca aleatória de hiperparâmetros com validação no treino benigno")
    ap.add_argument("--base", help="CSV de treino benigno usado na busca (padrão: 1º --treino)")
    ap.add_argument("--n-tentativas", type=int, default=8, help="nº de configurações testadas por (modelo, perda)")
    ap.add_argument("--val-frac", type=float, default=0.2, help="fração do treino benigno reservada p/ validação")
    ap.add_argument("--scoring", default="f1", choices=list(METRICAS), help="métrica usada para escolher a melhor config")
    ap.add_argument("--params", help="CSV de melhores parâmetros já encontrados (pula a busca)")
    ap.add_argument("--saida-params", default=str(SAIDA / "melhores_parametros_deep.csv"))
    ap.add_argument("--saida", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda")
    print("Dispositivo:", dev)
    testes = {nome_do_arquivo(p): carregar_csv(p) for p in args.teste}

    params = None
    if args.params:
        params = carregar_params(args.params)
        print(f"Usando parâmetros de {args.params}")
    elif args.otimizar:
        params, resumo, historico = {}, [], []
        base = carregar_csv(args.base or args.treino[0])
        print(f"Buscando hiperparâmetros em {args.base or args.treino[0]} ({len(base)} linhas)")
        for nome in args.modelos:
            for perda in args.perdas:
                melhor, score, hist = buscar_hiperparametros(nome, perda, base, args, dev)
                params[(nome, perda)] = melhor
                historico += hist
                resumo.append({"Modelo": nome, "Perda": perda, f"Val_{METRICAS[args.scoring]}": round(score, 4),
                               "Parametros": json.dumps(melhor)})
                pd.DataFrame(resumo).to_csv(args.saida_params, index=False)        # salva incrementalmente
                pd.DataFrame(historico).to_csv(str(args.saida_params).replace(".csv", "_historico.csv"), index=False)
                print(f"  >> melhor {nome}|{perda}: {melhor} ({METRICAS[args.scoring]}={score:.4f})")
        print(f"Melhores parâmetros salvos em {args.saida_params}")
    config = "otimizado" if params else "padrao"
    saida = args.saida or str(SAIDA / ("resultados_deep_otimizado.csv" if params else "resultados_deep.csv"))

    linhas = []
    for nome in args.modelos:
        for perda in args.perdas:
            hp = argparse.Namespace(**{**vars(args), **(params or {}).get((nome, perda), {})})
            for p_tr in args.treino:
                tr, nome_tr = carregar_csv(p_tr), nome_do_arquivo(p_tr)
                print(f"[{nome} | {perda} | treino={nome_tr}] n={len(tr)} lr={hp.lr} epocas={hp.epocas} "
                      f"batch={hp.batch} max_len={hp.max_len} tau={hp.tau}")
                res = rodar(nome, perda, tr, testes, hp, dev)
                for nome_te, r in res.items():
                    linhas.append({"Modelo": nome, "Perda": NOME_PERDA[perda], "Config": config,
                                   "Treino": nome_tr, "Teste": nome_te, **r})
                pd.DataFrame(linhas).to_csv(saida, index=False)   # salva incrementalmente
    salvar_resultados(pd.DataFrame(linhas), saida, index=["Treino", "Teste"], columns=["Modelo", "Perda"])


if __name__ == "__main__":
    main()
