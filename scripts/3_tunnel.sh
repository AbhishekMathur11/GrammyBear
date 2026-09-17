#!/usr/bin/env bash
# Terminal 3 — Cloudflare Tunnel. The public URL is reprinted in a big box.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
URL_FILE="${ROOT}/.tunnel_url"

announce() {
  local url="$1"
  printf '%s\n' "$url" >"${URL_FILE}"
  printf '\n\n'
  printf '==============================================================\n'
  printf '  TEDDYTALK PUBLIC URL — open this on another phone/laptop:\n'
  printf '\n'
  printf '  %s\n' "$url"
  printf '\n'
  printf '==============================================================\n'
  printf '  Also saved in: %s\n' "${URL_FILE}"
  printf '  Leave this terminal running. Ctrl+C closes the tunnel.\n'
  printf '\n\n'
}

echo "Tunneling http://localhost:8003 ..."
echo "Waiting for Cloudflare to assign a public URL..."
echo

found=0
ready=0
cloudflared tunnel --url http://localhost:8003 2>&1 | while IFS= read -r line; do
  printf '%s\n' "$line"
  if [[ "${found}" -eq 0 && "${line}" =~ (https://[a-zA-Z0-9.-]+\.trycloudflare\.com) ]]; then
    announce "${BASH_REMATCH[1]}"
    found=1
  fi
  if [[ "${ready}" -eq 0 && "${line}" == *"Registered tunnel connection"* ]]; then
    printf '\nTunnel is live. Do not close this terminal.\n\n'
    ready=1
  fi
done
