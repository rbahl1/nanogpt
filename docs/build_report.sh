#!/usr/bin/env bash
# REPORT.md -> REPORT.pdf (needs pandoc and Chrome/Chromium). Figures are embedded.
set -euo pipefail
cd "$(dirname "$0")/.."
pandoc REPORT.md -s --embed-resources --metadata lang=en -c docs/report.css -o docs/report.html
${CHROME:-google-chrome} --headless=new --no-sandbox --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$PWD/REPORT.pdf" "file://$PWD/docs/report.html" 2>/dev/null
rm docs/report.html
