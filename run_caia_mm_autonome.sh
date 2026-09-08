#!/usr/bin/env bash
# CAIA-MM unattended full-pipeline launcher + watchdog.
# Survives SSH disconnect (setsid). Restarts the orchestrator if it dies or hangs.
# Usage:
#   bash /home/mm-cfa-repro/run_caia_mm_autonome.sh          # detach
#   bash /home/mm-cfa-repro/run_caia_mm_autonome.sh --fg     # foreground (debug)
#   bash /home/mm-cfa-repro/run_caia_mm_autonome.sh --status
#   bash /home/mm-cfa-repro/run_caia_mm_autonome.sh --stop
set -u

ROOT="${MM_CFA_ROOT:-/home/mm-cfa-repro}"
export MM_CFA_ROOT="$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

LOGDIR="$ROOT/agent_extension/logs"
mkdir -p "$LOGDIR" "$ROOT/agent_extension/results" "$ROOT/agent_extension/traces"

PY="${CAIA_PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x /opt/miniconda3/envs/nutri/bin/python ]]; then
    PY=/opt/miniconda3/envs/nutri/bin/python
  else
    PY="$(command -v python3 || command -v python)"
  fi
fi
export CAIA_PYTHON="$PY"

WATCHDOG_PIDFILE="$LOGDIR/watchdog.pid"
ORCH_PIDFILE="$LOGDIR/orchestrator.pid"
HB="$LOGDIR/heartbeat.json"
COMPLETE="$LOGDIR/COMPLETE"
FATAL="$LOGDIR/FATAL"
WATCHLOG="$LOGDIR/watchdog.log"
ORCHLOG="$LOGDIR/orchestrator.log"
HANG_SEC="${CAIA_HANG_SEC:-1200}"
POLL_SEC="${CAIA_POLL_SEC:-20}"
MAX_RESTARTS="${CAIA_MAX_RESTARTS:-8}"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
wlog() { echo "[$(ts)] $*" | tee -a "$WATCHLOG"; }

is_alive() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

hb_age() {
  # Missing/unreadable heartbeat is "just started", not hung.
  if [[ ! -f "$HB" ]]; then
    echo 0
    return
  fi
  "$PY" -c 'import json,time,sys
p=sys.argv[1]
try:
    d=json.load(open(p))
    print(max(0, int(time.time()-float(d.get("unix", time.time())))))
except Exception:
    print(0)
' "$HB"
}

start_orch() {
  wlog "start orchestrator PY=$PY"
  # drop COMPLETE only if user asked for a fresh run
  if [[ "${CAIA_FRESH:-0}" == "1" ]]; then
    rm -f "$COMPLETE" "$FATAL"
    rm -f "$LOGDIR/pipeline_state.json"
  fi
  setsid "$PY" -u "$ROOT/agent_extension/run_pipeline.py" >>"$ORCHLOG" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$ORCH_PIDFILE"
  wlog "orchestrator pid=$pid"
}

cmd_status() {
  echo "MM_CFA_ROOT=$ROOT"
  echo "python=$PY"
  echo "watchdog_pid=$(cat "$WATCHDOG_PIDFILE" 2>/dev/null || echo none) alive=$(is_alive "$(cat "$WATCHDOG_PIDFILE" 2>/dev/null)" && echo yes || echo no)"
  echo "orch_pid=$(cat "$ORCH_PIDFILE" 2>/dev/null || echo none) alive=$(is_alive "$(cat "$ORCH_PIDFILE" 2>/dev/null)" && echo yes || echo no)"
  echo "complete=$([[ -f $COMPLETE ]] && echo yes || echo no)"
  echo "fatal=$([[ -f $FATAL ]] && echo yes || echo no)"
  if [[ -f "$HB" ]]; then
    echo "heartbeat:"
    cat "$HB"
  fi
  if [[ -f "$LOGDIR/pipeline_state.json" ]]; then
    echo "state:"
    "$PY" - <<'PY'
import json
from pathlib import Path
p=Path("/home/mm-cfa-repro/agent_extension/logs/pipeline_state.json")
d=json.loads(p.read_text())
for k,v in (d.get("steps") or {}).items():
    print(f"  {k:22s} {v.get('status')} rc={v.get('rc')} attempts={v.get('attempts')}")
print("degrade", d.get("degrade"))
PY
  fi
  echo "---- orchestrator tail ----"
  tail -n 30 "$ORCHLOG" 2>/dev/null || true
}

cmd_stop() {
  wlog "stop requested"
  touch "$LOGDIR/STOP"
  for f in "$ORCH_PIDFILE" "$WATCHDOG_PIDFILE" "$LOGDIR/e01_clean_rgb_rgbd/../results/e01_clean_rgb_rgbd/launched.json"; do
    :
  done
  if [[ -f "$ROOT/agent_extension/results/e01_clean_rgb_rgbd/launched.json" ]]; then
    e01="$("$PY" -c "import json; print(json.load(open('$ROOT/agent_extension/results/e01_clean_rgb_rgbd/launched.json')).get('pid',''))" 2>/dev/null || true)"
    if is_alive "$e01"; then
      wlog "stop e01 pid=$e01"
      kill "$e01" 2>/dev/null || true
    fi
  fi
  if [[ -f "$ORCH_PIDFILE" ]]; then
    opid="$(cat "$ORCH_PIDFILE")"
    if is_alive "$opid"; then
      kill "$opid" 2>/dev/null || true
      sleep 2
      kill -9 "$opid" 2>/dev/null || true
    fi
  fi
  if [[ -f "$WATCHDOG_PIDFILE" ]]; then
    wpid="$(cat "$WATCHDOG_PIDFILE")"
    if is_alive "$wpid"; then
      kill "$wpid" 2>/dev/null || true
    fi
  fi
  wlog "stopped"
}

watch_loop() {
  echo $$ > "$WATCHDOG_PIDFILE"
  wlog "watchdog start pid=$$ hang_sec=$HANG_SEC"
  local restarts=0
  local started_at
  start_orch
  started_at=$(date +%s)
  while true; do
    if [[ -f "$LOGDIR/STOP" ]]; then
      wlog "STOP flag — exiting watchdog"
      exit 0
    fi
    if [[ -f "$COMPLETE" ]]; then
      wlog "COMPLETE — watchdog exit 0"
      exit 0
    fi
    local opid=""
    [[ -f "$ORCH_PIDFILE" ]] && opid="$(cat "$ORCH_PIDFILE")"
    if ! is_alive "$opid"; then
      if [[ -f "$COMPLETE" ]]; then
        wlog "COMPLETE after process exit"
        exit 0
      fi
      restarts=$((restarts + 1))
      wlog "orchestrator dead — restart $restarts/$MAX_RESTARTS"
      if [[ "$restarts" -gt "$MAX_RESTARTS" ]]; then
        echo "$(ts) too many restarts" > "$FATAL"
        wlog "FATAL too many restarts"
        exit 1
      fi
      sleep 5
      start_orch
      started_at=$(date +%s)
      continue
    fi
    age="$(hb_age)"
    now=$(date +%s)
    grace=$((now - started_at))
    if [[ "$grace" -lt 90 ]]; then
      sleep "$POLL_SEC"
      continue
    fi
    if [[ "$age" -gt "$HANG_SEC" ]]; then
      wlog "heartbeat stale ${age}s > ${HANG_SEC}s — kill and restart"
      kill "$opid" 2>/dev/null || true
      sleep 3
      kill -9 "$opid" 2>/dev/null || true
      restarts=$((restarts + 1))
      if [[ "$restarts" -gt "$MAX_RESTARTS" ]]; then
        echo "$(ts) hung too many times" > "$FATAL"
        exit 1
      fi
      start_orch
    fi
    sleep "$POLL_SEC"
  done
}

case "${1:-}" in
  --status) cmd_status; exit 0 ;;
  --stop) cmd_stop; exit 0 ;;
  --fg)
    rm -f "$LOGDIR/STOP"
    watch_loop
    ;;
  *)
    rm -f "$LOGDIR/STOP"
    if is_alive "$(cat "$WATCHDOG_PIDFILE" 2>/dev/null || echo)"; then
      wlog "watchdog already running"
      cmd_status
      exit 0
    fi
    setsid bash "$0" --fg >>"$WATCHLOG" 2>&1 < /dev/null &
    echo $! > "$WATCHDOG_PIDFILE"
    sleep 1
    echo "started watchdog pid=$(cat "$WATCHDOG_PIDFILE")"
    echo "logs: $ORCHLOG"
    echo "status: bash $0 --status"
    ;;
esac
