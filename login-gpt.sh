#!/usr/bin/env bash
# Inicia el login local de ChatGPT/Codex OAuth desde la raíz del proyecto.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Igual que restart.sh: permite que GPT_CREDENTIAL_PATH coincida con el del bot.
if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

PYTHONPATH=src python3 -m oraculo.main auth login
