#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source scripts/load_local_env.sh
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
# Some desktop/shell proxy settings use the ambiguous socks:// scheme.
# requests needs an explicit version; use SOCKS5 with proxy-side DNS.
# Preserve each variable's precedence, endpoint and credentials; never print URLs.
for proxy_name in http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY; do
  proxy_value="${!proxy_name-}"
  if [[ "${proxy_value,,}" == socks://* ]]; then
    export "${proxy_name}=socks5h://${proxy_value#*://}"
    printf '[network] %s: socks:// -> socks5h://\n' "$proxy_name" >&2
  fi
done
unset proxy_name proxy_value
FR3_AGENT_PYTHON="${FR3_AGENT_PYTHON:-python3}"
# Stop cached camera streams when the interactive CLI exits, including on errors.
exec "$FR3_AGENT_PYTHON" -u -m app --close_services_on_exit "$@"
