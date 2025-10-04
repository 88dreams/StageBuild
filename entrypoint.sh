#!/usr/bin/env bash
set -e
# Activate venv only if it exists
[ -d /workspace/.venv ] && source /workspace/.venv/bin/activate || true
if [ $# -gt 0 ]; then
  exec "$@"
else
  exec /bin/bash
fi