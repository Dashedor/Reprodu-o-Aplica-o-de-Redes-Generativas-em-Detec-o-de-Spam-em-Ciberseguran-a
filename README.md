# Reprodução: Redes Generativas em Detecção de Spam (SBSeg 2025) — dataset YouTube

Todo CSV precisa ter as colunas `CONTENT` (texto) e `CLASS` (1 = Spam, 0 = Ham).

```bash
pip install -r requirements.txt

# 1) e 2) treinar/avaliar: informe os CSVs de treino e de teste (benignos ou envenenados)
python 02_modelos_classicos.py --treino saida/YT_Tr_Benigno.csv saida/YT_Tr_GPT-6.csv --teste saida/YT_Teste.csv
python 03_modelos_deep_learning.py --treino saida/YT_Tr_Benigno.csv saida/YT_Tr_GPT-6.csv --teste saida/YT_Teste.csv
```
Resultados salvos em CSV: `saida/resultados_classicos.csv` e `saida/resultados_deep.csv`

## Otimização de hiperparâmetros (opcional)
```bash
python 02_modelos_classicos.py --otimizar --treino saida/YT_Tr_Benigno.csv saida/YT_Tr_GPT-6.csv --teste saida/YT_Teste.csv
GridSearchCV (5 folds, métrica F1) só no treino benigno (1º `--treino`, ou `--base`). Os melhores parâmetros vão para
`saida/melhores_parametros.csv` e são aplicados em todos os treinos; resultados em `saida/resultados_classicos_otimizado*.csv`.
Para reaproveitar sem refazer a busca: `--params saida/melhores_parametros.csv`. Use `--cv 3` ou `--scoring roc_auc` para ajustar.

## Otimização de hiperparâmetros do BERT/RoBERTa (opcional)
```bash
python 03_modelos_deep_learning.py --otimizar --n-tentativas 8 --treino saida/YT_Tr_Benigno.csv saida/YT_Tr_GPT-6.csv --teste saida/YT_Teste.csv
```
Busca aleatória de `lr`, `epocas`, `batch`, `max_len` (e `tau` na perda Dual) para cada (modelo, perda), validando em 20% do
treino benigno (`--val-frac`). A config padrão é sempre uma das tentativas. Saídas: `saida/melhores_parametros_deep.csv`,
`saida/melhores_parametros_deep_historico.csv` (todas as tentativas) e `saida/resultados_deep_otimizado*.csv`.
Reuso sem refazer a busca: `--params saida/melhores_parametros_deep.csv`.

## RNNs (RNN, LSTM, BiLSTM, GRU, BiGRU)
`03_modelos_deep_learning.py` agora aceita, além de `BERT` e `RoBERTa`, as RNNs `RNN`, `LSTM`, `BiLSTM`, `GRU` e `BiGRU`
(PyTorch puro, treinadas do zero: vocabulário e embeddings aprendidos só com o CSV de treino).
```bash
python 03_modelos_deep_learning.py --modelos LSTM BiLSTM GRU --treino saida/YT_Tr_Benigno.csv saida/YT_Tr_GPT-6.csv --teste saida/YT_Teste.csv
python 03_modelos_deep_learning.py --modelos BiLSTM --otimizar --n-tentativas 10 --treino saida/YT_Tr_Benigno.csv saida/YT_Tr_GPT-6.csv --teste saida/YT_Teste.csv
```
Padrões das RNNs: `lr=1e-3`, 10 épocas, batch 32, embedding 128, 128 unidades, 1 camada, dropout 0,3 (opções `--embed`, `--hidden`,
`--camadas`, `--dropout`, `--min-freq`). Funcionam com Entropia Cruzada e com Dual Contrastive Learning, e com `--otimizar`/`--params`.
