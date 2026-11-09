# Reprodução: Redes Generativas em Detecção de Spam

Todo CSV precisa ter as colunas `CONTENT` (texto) e `CLASS` (1 = Spam, 0 = Ham).

```bash
pip install -r requirements.txt

# 0) (opcional) dividir o dataset original em treino/teste 70/30
python 00_dividir_dataset.py --csv YT_Dataset_.csv        # -> saida/treino.csv, saida/teste.csv

# 1) (opcional) gerar versões envenenadas; os spams reescritos SUBSTITUEM os originais e ganham rótulo 0
python 01_envenenar_csv.py --entrada saida/treino.csv --ref saida/teste.csv --backend gpt4o
python 01_envenenar_csv.py --entrada saida/treino.csv --ref saida/teste.csv --backend gemini
python 01_envenenar_csv.py --entrada saida/teste.csv --backend gpt4o      # teste envenenado (opcional)

# 2) e 3) treinar/avaliar: informe os CSVs de treino e de teste (benignos ou envenenados)
python 02_modelos_classicos.py     --treino saida/treino.csv saida/treino_gpt4o.csv saida/treino_gemini.csv --teste saida/teste.csv
python 03_modelos_deep_learning.py --treino saida/treino.csv saida/treino_gpt4o.csv saida/treino_gemini.csv --teste saida/teste.csv
```
Cada CSV de treino é combinado com cada CSV de teste. Resultados salvos em CSV: `saida/resultados_classicos.csv` e `saida/resultados_deep.csv` (formato longo) + `*_tabela.csv` (formato largo, como as tabelas do artigo).
`--backend local` roda sem API (reescrita por regras, só para testar o pipeline).
