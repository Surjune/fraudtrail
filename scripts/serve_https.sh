#!/usr/bin/env bash
# Serve the dashboard at https://DOMAIN on an Ubuntu server.
#
# Streamlit runs as a service on localhost, and Caddy sits in front of it on ports 80 and
# 443, obtaining and renewing the certificate itself. The server's plain IP address keeps
# working over http. Safe to run again after a git pull.
#
# usage: bash scripts/serve_https.sh fraudtrail.example.com
set -euo pipefail

DOMAIN="${1:?usage: bash scripts/serve_https.sh DOMAIN}"
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
UV="$HOME/.local/bin/uv"
PUBLIC_IP="$(curl -fsS https://checkip.amazonaws.com)"

echo "== installing the project"
cd "$APP_DIR"
"$UV" sync --quiet

echo "== running the dashboard as a service on localhost:8501"
sudo tee /etc/systemd/system/fraudtrail.service >/dev/null <<EOF
[Unit]
Description=FraudTrail dashboard
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$APP_DIR
ExecStart=$UV run streamlit run app/dashboard.py --server.address 127.0.0.1 --server.port 8501
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable fraudtrail >/dev/null
# Restarting first frees port 80, which an earlier version of this service held.
sudo systemctl restart fraudtrail

echo "== installing Caddy"
if ! command -v caddy >/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq caddy
fi

echo "== serving https://$DOMAIN and http://$PUBLIC_IP"
sudo tee /etc/caddy/Caddyfile >/dev/null <<EOF
$DOMAIN {
    reverse_proxy 127.0.0.1:8501
}

http://$PUBLIC_IP {
    reverse_proxy 127.0.0.1:8501
}
EOF
sudo systemctl enable caddy >/dev/null
sudo systemctl restart caddy

echo "== waiting for the certificate"
for _ in $(seq 1 30); do
  if curl -fsS -o /dev/null "https://$DOMAIN/_stcore/health"; then
    echo "done: https://$DOMAIN is live"
    exit 0
  fi
  sleep 4
done
echo "https://$DOMAIN is not answering yet. Check that the security group allows port 443,"
echo "then look at: sudo journalctl -u caddy -n 30 --no-pager"
exit 1
