#!/bin/bash
# prepare NOTE; test ROOT; run ROOT --max-parallel 2; mc ROOT; status ROOT; report ROOT
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$HERE/24_A_smallpanel_control.py" "$@"
