#!/bin/bash
# Publish the checkpoint and its card to a Hugging Face model repo.
#
#   .venv/bin/hf auth login              # once, with a write token
#   ./scripts/upload_model.sh <owner>/<name> /path/to/checkpoint
#
# Nothing runs without both arguments and an explicit confirmation. The card is uploaded from
# this repository so that what the Hub shows and what the code claims cannot drift apart.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="${1:?usage: $0 <owner>/<name> <checkpoint-dir>}"
CKPT="${2:?usage: $0 <owner>/<name> <checkpoint-dir>}"
CARD="${3:-../hf-model/README.md}"
HF=$(command -v hf || echo .venv/bin/hf)

for f in config.json model.safetensors tokenizer.json tokenizer_config.json; do
  [ -s "$CKPT/$f" ] || { echo "missing $CKPT/$f" >&2; exit 1; }
done
[ -s "$CARD" ] || { echo "no model card at $CARD" >&2; exit 1; }

WHO=$("$HF" auth whoami 2>/dev/null | head -1 || true)
case "$WHO" in ""|*"Not logged in"*)
  echo "not logged in: run  $HF auth login  with a write token first" >&2; exit 1 ;;
esac

SIZE=$(du -sh "$CKPT" | cut -f1)
echo "==> $REPO   (as $WHO)"
echo "    checkpoint $CKPT  ($SIZE)"
echo "    card       $CARD"
read -r -p "push to https://huggingface.co/$REPO ? [y/N] " ok
[ "$ok" = "y" ] || { echo "aborted"; exit 0; }

"$HF" repos create "$REPO" --repo-type model -y 2>/dev/null \
  || "$HF" repo create "$REPO" --repo-type model -y 2>/dev/null || true
"$HF" upload "$REPO" "$CARD" README.md >/dev/null
for f in config.json generation_config.json model.safetensors tokenizer.json \
         tokenizer_config.json chat_template.jinja; do
  [ -s "$CKPT/$f" ] || continue
  echo "    uploading $f"
  "$HF" upload "$REPO" "$CKPT/$f" "$f" >/dev/null
done
echo "==> https://huggingface.co/$REPO"
