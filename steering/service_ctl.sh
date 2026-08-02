#!/usr/bin/env bash
# Start/stop/health-check the steering service, detached so it survives the parent shell.
#
# Launching with a bare `nohup … &` from a tool call was not enough: when the caller was killed the
# service went with it, which would silently end an overnight run. `setsid` puts the service in its
# own session so nothing upstream can take it down.
#
#   steering/service_ctl.sh start      # start if not already running, wait until it answers
#   steering/service_ctl.sh status     # 0 = healthy (model loaded and /strategies answers)
#   steering/service_ctl.sh stop
#   steering/service_ctl.sh restart
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${STEER_PY:-/home/ubuntu/2616026/ES_approaches/env/bin/python}"
PORT="${STEER_PORT:-8100}"
URL="http://127.0.0.1:${PORT}"
LOG="$REPO/logs/service.log"
PIDFILE="$REPO/logs/service.pid"
PATTERN="uvicorn steering.serve_steer"

mkdir -p "$REPO/logs"

running() { pgrep -f "$PATTERN" >/dev/null 2>&1; }

healthy() {
  # /strategies triggers the lazy model load, so the first call can take minutes. Health means the
  # HTTP layer answers with the five strategies.
  curl -s -m "${1:-15}" "$URL/strategies" 2>/dev/null | grep -q "Self-disclosure"
}

start() {
  if running && healthy 10; then echo "already running and healthy"; return 0; fi
  running && { echo "process present but unhealthy — restarting"; stop; }
  echo "starting service (log: $LOG)"
  cd "$REPO" || exit 1
  setsid nohup "$PY" -m uvicorn steering.serve_steer:app \
      --host 127.0.0.1 --port "$PORT" >> "$LOG" 2>&1 < /dev/null &
  sleep 2
  pgrep -f "$PATTERN" | head -1 > "$PIDFILE"
  echo "  pid $(cat "$PIDFILE" 2>/dev/null); waiting for model load (up to 6 min)…"
  for _ in $(seq 1 72); do
    if healthy 20; then echo "  healthy"; return 0; fi
    running || { echo "  process died — see $LOG"; tail -5 "$LOG"; return 1; }
    sleep 5
  done
  echo "  TIMEOUT waiting for health"; tail -5 "$LOG"; return 1
}

stop() {
  pkill -f "$PATTERN" 2>/dev/null && echo "stopped" || echo "not running"
  rm -f "$PIDFILE"; sleep 2
}

case "${1:-status}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; start ;;
  status)
    if running && healthy 20; then echo "healthy (pid $(pgrep -f "$PATTERN" | head -1))"; exit 0
    elif running; then echo "running but not answering"; exit 1
    else echo "down"; exit 2; fi ;;
  *) echo "usage: $0 {start|stop|restart|status}"; exit 64 ;;
esac
