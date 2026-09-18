#!/usr/bin/env bash
# Terminal 3 — public HTTPS. Prefers https://teddytalk.loca.lt (named).
# Named Cloudflare tunnel (token) wins if CLOUDFLARED_TUNNEL_TOKEN is set.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
URL_FILE="${ROOT}/.tunnel_url"
CFG="${ROOT}/config.json"
TOKEN="${CLOUDFLARED_TUNNEL_TOKEN:-}"
FIXED="${TEDDYTALK_PUBLIC_URL:-}"
SUBDOMAIN="${TEDDYTALK_SUBDOMAIN:-}"

read_cfg() {
  python3 -c "import json; d=json.load(open('${CFG}')).get('tunnel',{}); print(d.get('$1','') or '')" 2>/dev/null || true
}

if [[ -z "${FIXED}" && -f "${CFG}" ]]; then
  FIXED=$(read_cfg public_url)
fi
if [[ -z "${SUBDOMAIN}" && -f "${CFG}" ]]; then
  SUBDOMAIN=$(read_cfg subdomain)
fi
SUBDOMAIN="${SUBDOMAIN:-teddytalk}"
FIXED="${FIXED:-https://${SUBDOMAIN}.loca.lt}"

LAN=$(hostname -I 2>/dev/null | awk '{print $1}')

announce() {
  local url="$1"
  printf '%s\n' "$url" >"${URL_FILE}"
  printf '\n\n'
  printf '==============================================================\n'
  printf '  TEDDYTALK PUBLIC URL — open this on any phone/laptop:\n'
  printf '\n'
  printf '  %s\n' "$url"
  printf '\n'
  if [[ -n "${LAN}" ]]; then
    printf '  Same Wi-Fi only (mic often blocked on HTTP):\n'
    printf '  http://%s:8003\n' "${LAN}"
    printf '\n'
  fi
  printf '==============================================================\n'
  printf '  Also saved in: %s\n' "${URL_FILE}"
  printf '  Leave this terminal running. Ctrl+C closes the tunnel.\n'
  printf '\n\n'
}

echo "Tunneling http://localhost:8003 ..."
echo

if [[ -n "${TOKEN}" ]]; then
  announce "${FIXED}"
  exec cloudflared tunnel run --token "${TOKEN}"
fi

if command -v npx >/dev/null 2>&1; then
  echo "Publishing a named URL: ${FIXED}"
  echo "If loca.lt shows a click-through page on the phone, tap Continue once."
  echo
  announce "${FIXED}"
  exec npx --yes localtunnel --port 8003 --subdomain "${SUBDOMAIN}"
fi

echo "npx not found — falling back to a random trycloudflare.com URL."
echo "Waiting for Cloudflare..."
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
