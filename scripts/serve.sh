#!/usr/bin/env bash
# launchd wrapper for the TGN search API.
#
# serve.py doesn't strictly need LM Studio to *start* (it only calls it
# per-query), but launchd has no native "depend on another service" for a
# GUI app like LM Studio. So we approximate the dependency here:
#   1. best-effort nudge LM Studio's server up (idempotent), then
#   2. block until :1234 answers, so the API comes up clean instead of
#      erroring on early queries.
# KeepAlive in the plist handles crash/exit restarts.
set -u

# Repo root = parent of this script's dir, regardless of how launchd invokes it.
cd "$(cd "$(dirname "$0")/.." && pwd)" || exit 1

LMS="$HOME/.lmstudio/bin/lms"
LM_URL="${LM_STUDIO_URL:-http://127.0.0.1:1234}"

# Best-effort: ask LM Studio to start its server (no-op if already running).
if [ -x "$LMS" ]; then
  "$LMS" server start >/dev/null 2>&1 || true
fi

# Wait (up to ~120s) for LM Studio to answer before serving.
echo "$(date '+%F %T') waiting for LM Studio at $LM_URL ..."
for i in $(seq 1 120); do
  if curl -sf -m 3 "$LM_URL/v1/models" >/dev/null 2>&1; then
    echo "$(date '+%F %T') LM Studio is up (after ${i}s) — starting search API"
    break
  fi
  sleep 1
done

exec ./venv/bin/python web/serve.py
