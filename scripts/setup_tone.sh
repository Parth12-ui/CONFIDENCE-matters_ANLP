#!/usr/bin/env bash
# Local CPU setup for the tone-severity scoring (src/tone/): a venv with the scoring
# packages, the NRC-VAD lexicon and the GoEmotions classifier. The ConvoKit politeness
# corpus downloads on first use. Run from the repository root.
#   bash scripts/setup_tone.sh
# Both downloads can be slow from some networks; fetching them on a faster machine (e.g. the
# Ada login node) and copying the files into resources/ works the same.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -x .venv_tone/bin/python ] || python3 -m venv .venv_tone
.venv_tone/bin/pip install -q --upgrade pip
.venv_tone/bin/pip install -q torch transformers convokit scipy pandas scikit-learn
.venv_tone/bin/python -m spacy download en_core_web_sm

mkdir -p resources
# NRC-VAD (Mohammad, ACL 2018): free for research use, cite the paper; not redistributed here
if [ ! -f resources/NRC-VAD-Lexicon.txt ]; then
  curl -fL --retry 5 -C - -o resources/NRC-VAD-Lexicon.zip https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon.zip
  unzip -p resources/NRC-VAD-Lexicon.zip NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt > resources/NRC-VAD-Lexicon.txt
fi
wc -l resources/NRC-VAD-Lexicon.txt

# GoEmotions classifier (RoBERTa-base fine-tuned on GoEmotions, Demszky et al., ACL 2020)
mkdir -p resources/goemotions
for f in config.json model.safetensors tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json; do
  [ -f resources/goemotions/$f ] || curl -fL --retry 5 -o resources/goemotions/$f \
    https://huggingface.co/SamLowe/roberta-base-go_emotions/resolve/main/$f
done
ls -la resources/goemotions
