#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export PATH="/root/.local/bin:${PATH}"

if ! command -v popcorn-cli >/dev/null 2>&1; then
  echo "error: popcorn-cli not found. Install it or add it to PATH." >&2
  exit 1
fi

popcorn-cli submit \
  --gpu B200 \
  --leaderboard qr_v2 \
  --mode leaderboard \
  --no-tui \
  submission.py
