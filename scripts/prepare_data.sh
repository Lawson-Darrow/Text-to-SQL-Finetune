#!/usr/bin/env bash
# Milestone 3 data setup — run in WSL2. Fetches a self-consistent Spider package
# (166 SQLite DBs + tables.json + dev.json + dev_gold.sql) and the official evaluator
# into $T2S_HOME (default ~/t2s-data). Lives outside the repo (large, regenerable).
#
#   bash <(sed 's/\r$//' /mnt/c/Users/lawso/Desktop/Projects/Text-to-SQL-Finetune/scripts/prepare_data.sh)
set -euo pipefail

T2S_HOME="${T2S_HOME:-$HOME/t2s-data}"
mkdir -p "$T2S_HOME"; cd "$T2S_HOME"
source "$HOME/t2s-derisk/.venv/bin/activate"
uv pip install -q func_timeout nltk sqlparse  # evaluator deps
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Spider DBs + tables.json + dev.json + dev_gold.sql (one self-consistent package).
if [ ! -d data/database ]; then
  python -c "from huggingface_hub import hf_hub_download; hf_hub_download('SALT-NLP/spider_VALUE','data.zip',repo_type='dataset',local_dir='.')"
  python -c "import zipfile; zipfile.ZipFile('data.zip').extractall('.')"
fi
echo "DBs: $(ls data/database | wc -l) | tables.json: $(ls -lh data/tables.json | awk '{print $5}') | dev: $(python -c "import json;print(len(json.load(open('data/dev.json'))))")"

# Official Spider / test-suite evaluator (taoyds/test-suite-sql-eval).
[ -d test-suite-sql-eval ] || git clone --depth 1 https://github.com/taoyds/test-suite-sql-eval.git

echo "data ready at $T2S_HOME (set T2S_DATA=$T2S_HOME/data, T2S_EVAL=$T2S_HOME/test-suite-sql-eval)"
