#!/usr/bin/env bash
# Download every file of a Hugging Face model repo with curl (resumable), on the login node.
# Usage: bash scripts/fetch_model.sh ORG/NAME DEST_DIR [TOKEN_FILE]
# Run on the Ada login node: compute nodes download very slowly. Pass DEST_DIR to the
# run scripts as MODEL (e.g. MODEL=$HOME/models/Qwen2.5-7B-Instruct), or keep it in /share1
# and pass MODEL_ON_LOGIN. /share1 allows 25 GB per user: one 7-8B model at a time; a full
# quota shows up as curl "Failure writing output to destination".
set -uo pipefail
REPO="$1"; DEST="$2"; TOK="${3:-}"
AUTH=(); [ -n "$TOK" ] && [ -s "$TOK" ] && AUTH=(-H "Authorization: Bearer $(cat "$TOK")")
mkdir -p "$DEST"
FILES=$(curl -sfL "${AUTH[@]}" "https://huggingface.co/api/models/$REPO" | python3 -c "import sys,json; print(\"\n\".join(s[\"rfilename\"] for s in json.load(sys.stdin)[\"siblings\"] if not s[\"rfilename\"].startswith(\"original/\") and not s[\"rfilename\"].endswith(\".pth\")))")
[ -z "$FILES" ] && { echo "no file list (gated without access?)"; exit 1; }
for f in $FILES; do
  mkdir -p "$DEST/$(dirname "$f")"
  for try in 1 2 3 4 5; do
    curl -sfL "${AUTH[@]}" -C - --speed-limit 100000 --speed-time 60 -o "$DEST/$f" "https://huggingface.co/$REPO/resolve/main/$f" && break
    echo "retry $try: $f"; sleep 10
  done
done
echo "DONE $REPO -> $DEST"
