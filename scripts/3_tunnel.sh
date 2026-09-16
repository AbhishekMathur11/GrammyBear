#!/usr/bin/env bash
# Terminal 3 — Cloudflare Tunnel. Use the https:// URL it prints (mic needs HTTPS).
set -euo pipefail
exec cloudflared tunnel --url http://localhost:8003
