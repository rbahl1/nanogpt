#!/usr/bin/env bash
# REPORT.md -> REPORT.pdf (needs pandoc and Chrome/Chromium).
set -euo pipefail
cd "$(dirname "$0")"
pandoc ../REPORT.md -s --metadata lang=en -c report.css -o report.html
${CHROME:-google-chrome} --headless=new --no-sandbox --disable-gpu --no-pdf-header-footer --print-to-pdf=../REPORT.pdf report.html
rm report.html
