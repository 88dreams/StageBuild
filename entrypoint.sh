#!/usr/bin/env bash
set -e
source /workspace/.venv/bin/activate
if [ $# -gt 0 ]; then
  exec "$@"
else
  exec /bin/bash
fi