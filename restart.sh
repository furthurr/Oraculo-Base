#!/usr/bin/env bash
# restart.sh — Detiene y arranca Oraculo en segundo plano (modo polling).
#
# Uso:
#   ./restart.sh          # detiene cualquier instancia y arranca una nueva
#   ./restart.sh stop     # solo detiene
#   ./restart.sh status   # muestra si está corriendo
#
# El proceso queda en segundo plano (nohup) y escribe en oraculo.log.
# Al reiniciar se pierden las sesiones en memoria (por diseño); para
# reiniciar solo una conversación usa /new en el chat.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ -x "$ROOT/.venv312/bin/python" ]; then
  DEFAULT_PYTHON="$ROOT/.venv312/bin/python"
else
  DEFAULT_PYTHON="$ROOT/.venv/bin/python"
fi
PYTHON="${ORACULO_PYTHON:-$DEFAULT_PYTHON}"
LOG="$ROOT/oraculo.log"
# Coincide con las formas habituales de lanzar el bot sin capturar el propio
# script, pgrep/pkill ni otros procesos cuyo nombre solo contenga ese texto.
BOT_PATTERN='(^|[[:space:]])-m[[:space:]]+oraculo\.main([[:space:]]|$)|(^|[[:space:]])[^[:space:]]*/oraculo/main\.py([[:space:]]|$)'

bot_pids() {
  pgrep -f "$BOT_PATTERN" || true
}

signal_bot_pids() {
  local signal="$1" pids="$2" pid
  while IFS= read -r pid; do
    [ -n "$pid" ] || continue
    kill -s "$signal" "$pid" 2>/dev/null || true
  done <<< "$pids"
}

stop_bot() {
  local pids
  pids="$(bot_pids)"
  if [ -z "$pids" ]; then
    echo "No había ninguna instancia corriendo."
    return 0
  fi

  echo "Deteniendo todas las instancias de Oraculo..."
  for _ in $(seq 1 10); do
    pids="$(bot_pids)"
    [ -z "$pids" ] && break
    signal_bot_pids TERM "$pids"
    sleep 0.5
  done

  pids="$(bot_pids)"
  if [ -n "$pids" ]; then
    echo "Hay procesos que no terminaron con SIGTERM; forzando (SIGKILL)..."
    signal_bot_pids KILL "$pids"
    for _ in $(seq 1 10); do
      pids="$(bot_pids)"
      [ -z "$pids" ] && break
      signal_bot_pids KILL "$pids"
      sleep 0.2
    done
  fi

  pids="$(bot_pids)"
  if [ -n "$pids" ]; then
    echo "ERROR: siguen activas instancias de Oraculo; no se iniciará otra." >&2
    printf '%s\n' "$pids" >&2
    return 1
  fi
  echo "Todas las instancias están detenidas."
}

status_bot() {
  local pids
  pids="$(bot_pids)"
  if [ -n "$pids" ]; then
    pgrep -fl "$BOT_PATTERN"
    return 0
  fi
  echo "Oraculo no está corriendo."
}

start_bot() {
  [ -x "$PYTHON" ] || { echo "ERROR: no existe $PYTHON (¿falta crear el venv?)"; exit 1; }
  [ -f "$ROOT/.env" ] || { echo "ERROR: falta $ROOT/.env (copia .env.example y rellénalo)"; exit 1; }

  echo "Arrancando Oraculo..."
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
  PYTHONPATH="$ROOT/src" nohup "$PYTHON" -m oraculo.main >> "$LOG" 2>&1 &
  local pid=$!

  sleep 3
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "ERROR: el proceso terminó al arrancar. Últimas líneas del log:"
    tail -n 15 "$LOG"
    exit 1
  fi
  echo "Oraculo corriendo (PID $pid). Log: $LOG"
  tail -n 3 "$LOG"
}

case "${1:-restart}" in
  restart) stop_bot; start_bot ;;
  stop)    stop_bot ;;
  status)  status_bot ;;
  *)
    echo "Uso: $0 [stop|status]"
    exit 1
    ;;
esac
