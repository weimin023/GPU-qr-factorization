#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

source ./cutedsl_env.sh
python profile/verify_submission.py --module submission --trials 3 --benchmark-torch
