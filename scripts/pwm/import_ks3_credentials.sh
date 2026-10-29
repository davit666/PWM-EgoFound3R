#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
umask 077
mkdir -p outputs/secrets
chmod 700 outputs/secrets
trap 'stty echo' EXIT
stty -echo
python3 scripts/pwm/import_ks3_credentials.py
