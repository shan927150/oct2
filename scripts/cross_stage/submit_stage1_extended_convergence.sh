#!/bin/bash
# Compatibility entry retained for the command distributed in the earlier handoff.
# The reviewed implementation lives in submit_A_step1.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/submit_A_step1.sh" "$@"
