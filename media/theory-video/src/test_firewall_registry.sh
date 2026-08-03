#!/usr/bin/env bash
set -euo pipefail

# Generates real Socket Firewall traffic so a "live proof" video shows varied
# alert types actually landing in the Events page, not a narrated diagram.
#
# Every default package is a verified-safe test package (never real malware —
# see docs.socket.dev/docs/sample-malware-packages). Run this before recording;
# see SKILL.md "Testing Socket Firewall" for full findings, verified alerts per
# package, and known product gaps.
#
# Modes:
#   registry (default) — Socket's official registry-mode Docker image, run
#     locally. Real Warn + Monitor enforcement, npm + pypi. Needs docker.
#   wrapper — local Enterprise binary. Real Block enforcement, npm + Critical
#     CVE only, but has shown intermittent flakiness — see SKILL.md. Needs
#     SOCKET_API_KEY with security-policy:read.
#   manifest — one package.json listing every PATTERNS package, one
#     `sfw npm install`, simulating a real developer install.
#
# Usage: test_firewall_registry.sh [wrapper|registry|manifest] [--package <name>]

MODE="registry"
CUSTOM_PACKAGE=""
# Exactly one mode argument: sequential ifs would let "registry wrapper" consume
# both and silently run the LAST one.
case "${1:-}" in
  registry|wrapper|manifest) MODE="$1"; shift;;
esac
if [[ "${1:-}" == "--package" ]]; then CUSTOM_PACKAGE="${2:-}"; shift 2 || true; fi
# A typo'd mode ("Wrapper", "registryy") must not silently fall back to
# registry mode — the operator would believe they exercised a different mode.
if [[ -n "${1:-}" ]]; then
  echo "Unknown argument: '$1'. Usage: test_firewall_registry.sh [wrapper|registry|manifest] [--package <name>]" >&2
  exit 2
fi
if [[ -n "$CUSTOM_PACKAGE" && "$MODE" == "registry" ]]; then
  echo "--package is not supported in registry mode yet (it always sends its own" >&2
  echo "fixed REGISTRY_PATTERNS) — ignoring --package $CUSTOM_PACKAGE." >&2
  CUSTOM_PACKAGE=""
fi

# Hard safety gate — never install a real malware package, even via --package.
# Names pulled directly from docs.socket.dev/docs/sample-malware-packages' "already
# removed" section (real confirmed malware, explicit do-not-install warning there).
# This is a floor, not a substitute for judgment: it only catches these exact known
# names, not anything novel — never add a package here without checking that page.
MALWARE_DENYLIST=(
  webb3 filebdecoder node-click deahub litter-woker pyautodllxd koja_ali_jutt
  mrp-component-icon orange_papaya_greed 123rf-ui-core namatnawbyteweb1
  vue3-babel-js namatnawbyteweb6 airbnb-dls-web js-integration-demo
  segment-bundle dvuln
)
if [[ -n "$CUSTOM_PACKAGE" ]]; then
  # Strip a leading @scope/ (scoped package) then an @version, case-fold via tr
  # (bash 3.2 on stock macOS has no ${var,,}) before comparing against the denylist.
  CUSTOM_PACKAGE_NAME="$(echo "$CUSTOM_PACKAGE" | sed -E 's#^(@[^/]+/[^@]+|[^@]+).*#\1#' | tr '[:upper:]' '[:lower:]')"
  for bad in "${MALWARE_DENYLIST[@]}"; do
    if [[ "$CUSTOM_PACKAGE_NAME" == "$(echo "$bad" | tr '[:upper:]' '[:lower:]')" ]]; then
      echo "Refusing: '$CUSTOM_PACKAGE_NAME' is real, confirmed malware per" >&2
      echo "docs.socket.dev/docs/sample-malware-packages — never install it, even for" >&2
      echo "testing. Use one of this script's default patterns instead." >&2
      exit 1
    fi
  done
fi

# name|package spec|what it demonstrates|expected action per this org's Security
# Policy (Settings > Alerts > Security Policy — edit to match a different org).
# Each alert is verified against the package's own live Socket alerts page, not
# assumed from the name. Only minimist's Critical CVE actually enforces (Block) in
# wrapper mode — see SKILL.md for the full enforcement-gap findings.
# Ignore-tier and Block-tier only — Warn/Monitor live in REGISTRY_PATTERNS below,
# since wrapper mode reliably tags Ignore (and, when not flaky, Block) but registry
# mode doesn't even log an event for genuinely clean packages.
PATTERNS=(
  "tslib|tslib@2.6.2|verified zero alerts|Ignore (no alert)"
  "ms|ms@2.0.0|verified zero alerts|Ignore (no alert)"
  "chalk|chalk@5.3.0|verified zero alerts|Ignore (no alert)"
  "supports-color|supports-color@9.4.0|verified zero alerts|Ignore (no alert)"
  "camelcase|camelcase@8.0.0|verified zero alerts|Ignore (no alert)"
  "p-limit|p-limit@5.0.0|verified zero alerts|Ignore (no alert)"
  "is-plain-obj|is-plain-obj@4.1.0|verified zero alerts|Ignore (no alert)"
  "mri|mri@1.2.0|verified zero alerts|Ignore (no alert)"
  "sisteransi|sisteransi@1.0.5|verified zero alerts|Ignore (no alert)"
  "strip-ansi|strip-ansi@7.1.0|verified zero alerts|Ignore (no alert)"
  "minimist|minimist@0.0.5|verified live as Critical CVE|Block (Critical CVE)"
  "minimist-1.2.5|minimist@1.2.5|verified live as Critical CVE|Block (Critical CVE)"
  "momentjs|momentjs@2.0.0|verified typosquat of 'moment' (an npm collaborator-published, 187-byte inert stub — no code, just a README pointing to moment) — flagged live as 'AI-detected possible typosquat'|Block (Possible typosquat attack)"
  "deep-extend|deep-extend@0.4.0|verified live as Critical CVE|Block (Critical CVE)"
  "node-serialize|node-serialize@0.0.4|verified live as Critical CVE|Block (Critical CVE)"
  "vm2|vm2@3.9.10|verified live as Critical CVE|Block (Critical CVE)"
  "growl|growl@1.9.2|verified live as Critical CVE|Block (Critical CVE)"
  "uglify-js|uglify-js@2.4.0|verified live as Critical CVE|Block (Critical CVE)"
  "handlebars|handlebars@4.0.5|verified live as Critical CVE|Block (Critical CVE)"
  "mongoose|mongoose@4.13.6|verified live as Critical CVE|Block (Critical CVE)"
  "sequelize|sequelize@3.0.0|verified live as Critical CVE|Block (Critical CVE)"
  "jsonwebtoken|jsonwebtoken@0.4.0|verified live as Critical CVE|Block (Critical CVE)"
  "ejs|ejs@2.5.5|verified live as Critical CVE|Block (Critical CVE)"
)
[[ -n "$CUSTOM_PACKAGE" ]] && PATTERNS=("custom|$CUSTOM_PACKAGE|custom package requested via --package|unknown — check org policy")

# Shared by run_wrapper/run_manifest_batch: ensures SOCKET_API_KEY + sfw binary,
# then runs a security-policy:read canary check (see SKILL.md for why it's needed).
ensure_sfw_ready() {
  if [[ -z "${SOCKET_API_KEY:-}" ]]; then
    echo "SOCKET_API_KEY is not set." >&2
    echo "Create one INSIDE the org whose Events page you want this traffic to land" >&2
    echo "in — scopes: packages, entitlements:list, AND security-policy:read (the last" >&2
    echo "one is undocumented but required for real Block/Warn enforcement, not just" >&2
    echo "packages/entitlements:list — this script's own preflight canary check will" >&2
    echo "warn you if it looks missing). Then: export SOCKET_API_KEY=<key>" >&2
    exit 1
  fi

  SFW_HOME="${SFW_HOME:-$HOME/.local/share/socket-demo-video/sfw-enterprise}"
  if [[ ! -x "$SFW_HOME/sfw" ]]; then
    local os arch asset
    os="$(uname -s)"; arch="$(uname -m)"
    case "$os-$arch" in
      Darwin-arm64) asset=sfw-macos-arm64 ;;
      Darwin-x86_64) asset=sfw-macos-x86_64 ;;
      Linux-aarch64|Linux-arm64) asset=sfw-linux-arm64 ;;
      Linux-x86_64) asset=sfw-linux-x86_64 ;;
      *) echo "Unsupported platform $os-$arch — download manually from" >&2
         echo "https://github.com/SocketDev/firewall-release/releases" >&2
         exit 1 ;;
    esac
    echo "Downloading Socket Firewall Enterprise ($asset) to $SFW_HOME ..."
    mkdir -p "$SFW_HOME"
    # -f: a 404/renamed asset must fail here, not cache an HTML error page as an
    # executable that breaks every later run.
    curl -fsL "https://github.com/SocketDev/firewall-release/releases/latest/download/$asset" -o "$SFW_HOME/sfw"
    chmod +x "$SFW_HOME/sfw"
    [[ "$os" == "Darwin" ]] && xattr -dr com.apple.quarantine "$SFW_HOME/sfw" 2>/dev/null || true
  fi
  export PATH="$SFW_HOME:$PATH"
  echo "Using $(sfw --version 2>&1 | head -1)."
  echo

  # minimist@0.0.5 is a real, safe Critical CVE (CVE-2021-44906) used only as a
  # scope-detection canary: without security-policy:read it installs cleanly.
  # Heavy reuse of this exact version across a long session can go stale (cached
  # as "allow") — if this warns unexpectedly, try a different 0.0.x version.
  echo "Preflight: checking SOCKET_API_KEY's token can see the org's Security Policy..."
  local canary_dir canary_status canary_log
  canary_dir="$(mktemp -d)"
  canary_log="$(mktemp)"
  set +e
  ( cd "$canary_dir" && npm init -y >/dev/null 2>&1 && sfw npm install minimist@0.0.5 --no-audit --no-fund ) >"$canary_log" 2>&1
  canary_status=$?
  set -e
  rm -rf "$canary_dir"
  # A non-zero exit alone isn't proof of a real block — a network hiccup or registry
  # error would also fail here. Only trust the log's own evidence of a policy refusal
  # (npm's 403 + "forbidden by your security policy" text on an actual Firewall block).
  if [[ $canary_status -ne 0 ]] && grep -qi "forbidden by your security policy" "$canary_log"; then
    echo "Preflight OK: canary package was blocked as expected — this token can see the"
    echo "org's Security Policy. Block enforcement for Critical CVEs is confirmed working."
    rm -f "$canary_log"
  else
    echo "WARNING: the canary package (minimist@0.0.5, a confirmed Critical CVE) did not" >&2
    echo "get blocked by Socket's security policy. If it installed cleanly (exit 0), this" >&2
    echo "almost always means the token behind SOCKET_API_KEY is missing the" >&2
    echo "'security-policy:read' scope — without it, wrapper mode cannot see the org's" >&2
    echo "Security Policy and every install below will land in Events with alertAction" >&2
    echo "\"ignore\", regardless of severity. Fix: Socket dashboard -> Settings -> API" >&2
    echo "Tokens -> (your token) -> Edit scopes -> check 'security-policy:read' ->" >&2
    echo "Confirm. If it failed for a different reason (exit $canary_status, no security-policy" >&2
    echo "refusal message), that's a network/registry issue, not a policy finding —" >&2
    echo "see $canary_log for the actual error. Continuing anyway, but expect no real" >&2
    echo "Block/Warn distinction in the results below." >&2
  fi
  echo
}

run_wrapper() {
  ensure_sfw_ready
  echo "Sending varied test traffic through Socket Firewall Enterprise (wrapper mode, local)."
  echo

  for entry in "${PATTERNS[@]}"; do
    IFS='|' read -r label pkg demonstrates expected <<< "$entry"
    echo "--- $label ---"
    echo "demonstrates: $demonstrates"
    echo "expected action (per org policy — only Critical-CVE Block is confirmed to actually enforce; see SKILL.md): $expected"
    local tmpdir; tmpdir="$(mktemp -d)"
    # A Block-tier package is EXPECTED to make npm exit non-zero — that's the
    # demo working, not a script failure. Never let set -e kill the loop here.
    set +e
    ( cd "$tmpdir" && npm init -y >/dev/null 2>&1 && sfw npm install "$pkg" --no-audit --no-fund )
    local install_status=$?
    set -e
    [[ $install_status -ne 0 ]] && echo "(install exited $install_status — expected for Block-tier patterns)"
    rm -rf "$tmpdir"
    echo
  done
}

# One package.json listing every PATTERNS package, one `sfw npm install` —
# simulates a developer cloning a repo and installing once, instead of N
# separate throwaway installs.
run_manifest_batch() {
  ensure_sfw_ready
  echo "Building one realistic package.json from every PATTERNS entry, then running a"
  echo "single 'sfw npm install' — simulating a developer cloning a repo and running"
  echo "install once, with every dependency checked together in one batch."
  echo

  local manifest_dir; manifest_dir="$(mktemp -d)"
  (
    cd "$manifest_dir"
    npm init -y >/dev/null 2>&1
    node -e '
      const fs = require("fs");
      const patterns = process.argv.slice(1);
      const pkg = JSON.parse(fs.readFileSync("package.json", "utf8"));
      pkg.name = "sfw-manifest-demo";
      pkg.dependencies = {};
      for (const entry of patterns) {
        const [, spec] = entry.split("|");
        const at = spec.lastIndexOf("@");
        const name = at > 0 ? spec.slice(0, at) : spec;
        const version = at > 0 ? spec.slice(at + 1) : "*";
        if (pkg.dependencies[name] === undefined) {
          pkg.dependencies[name] = version;
        } else {
          // Same package at two versions (e.g. minimist@0.0.5 AND @1.2.5):
          // a plain key would silently drop one — install both via npm alias.
          const alias = `${name}-v${version.replace(/[^\w.]+/g, "-")}`;
          pkg.dependencies[alias] = `npm:${name}@${version}`;
        }
      }
      fs.writeFileSync("package.json", JSON.stringify(pkg, null, 2));
    ' "${PATTERNS[@]}"
    echo "package.json dependencies for this batch:"
    node -e 'console.log(JSON.stringify(JSON.parse(require("fs").readFileSync("package.json","utf8")).dependencies, null, 2))'
    echo
    # Blocks are the expected outcome for several entries — don't let set -e
    # abort before the Events-page instructions print.
    sfw npm install --no-audit --no-fund || echo "(batch install exited non-zero — expected when Block-tier patterns are enforced)"
  )
  rm -rf "$manifest_dir"
  echo
}

# Verified live against the official registry-mode Docker image — produces real,
# distinct Warn/Monitor (unlike wrapper mode). Even Critical CVE only reaches Warn
# here, never Block — see SKILL.md. ecosystem|package|version|route|demonstrates|expected
REGISTRY_PATTERNS=(
  "npm|minimist|0.0.5|npm|Critical CVE|Warn (registry mode's ceiling — not Block, even for Critical CVE)"
  "npm|trim-newlines|3.0.0|npm|High CVE|Warn"
  "npm|qs|6.2.0|npm|High + Medium CVE|Warn"
  "npm|negotiator|0.5.3|npm|High CVE|Warn"
  "npm|y18n|3.2.0|npm|High CVE|Warn"
  "npm|trim|0.0.1|npm|High CVE|Warn"
  "npm|debug|2.2.0|npm|High CVE|Warn"
  "npm|semver|5.1.0|npm|High CVE|Warn"
  "npm|braces|1.8.5|npm|High CVE|Warn"
  "npm|minimatch|0.2.14|npm|High CVE|Warn"
  "npm|axios|0.21.0|npm|High + Medium CVE|Warn"
  "npm|momentjs|2.0.0|npm|typosquat of moment|Warn"
  "npm|vuejs|3.0.1|npm|typosquat of vue|Warn"
  "npm|request|2.88.2|npm|Deprecated|Warn"
  "npm|peacenotwar|9.1.7|npm|Protestware or potentially unwanted behavior (the real 2022 node-ipc incident's companion package)|Warn"
  "npm|extend|3.0.0|npm|Medium CVE|Monitor"
  "npm|hosted-git-info|2.7.0|npm|Medium CVE|Monitor"
  "npm|tough-cookie|2.4.2|npm|Medium CVE|Monitor"
  "npm|nanoid|5.0.4|npm|Medium CVE|Monitor"
  "npm|uuid|3.3.0|npm|Medium CVE + Deprecated (Deprecated outweighs Medium CVE)|Warn"
  "npm|socket.io|1.0.0|npm|Medium CVE|Monitor"
  "npm|got|11.8.2|npm|Medium CVE|Monitor"
  "pypi|urllib3|1.24.1|pypi|High CVE|Warn"
  "pypi|light-s3-client|0.0.11|pypi|Unpopular package (Quality, not a CVE)|Monitor"
  "pypi|idna|2.7|pypi|Medium CVE|Monitor"
)

REGISTRY_HOME="${REGISTRY_HOME:-$HOME/.local/share/socket-demo-video/registry-firewall}"
REGISTRY_HTTPS_PORT="${REGISTRY_HTTPS_PORT:-18443}"
REGISTRY_CONTAINER="${REGISTRY_CONTAINER:-socket-registry-firewall-demo}"

# Starts (or reuses) a local instance of Socket's registry-mode Docker image.
ensure_local_registry() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required for the 'registry' test mode (it runs Socket's own" >&2
    echo "registry-mode image locally) — install Docker, or use 'wrapper' mode instead." >&2
    exit 1
  fi
  if [[ -z "${SOCKET_API_KEY:-}" ]]; then
    echo "SOCKET_API_KEY is not set (used as SOCKET_SECURITY_API_TOKEN for the registry" >&2
    echo "container — same key as wrapper mode). export SOCKET_API_KEY=<key>" >&2
    exit 1
  fi

  mkdir -p "$REGISTRY_HOME/ssl"
  if [[ ! -f "$REGISTRY_HOME/ssl/fullchain.pem" ]]; then
    openssl req -x509 -newkey rsa:2048 -keyout "$REGISTRY_HOME/ssl/privkey.pem" \
      -out "$REGISTRY_HOME/ssl/fullchain.pem" -days 365 -nodes -subj "/CN=localhost" \
      >/dev/null 2>&1
  fi
  cat > "$REGISTRY_HOME/socket.yml" <<'YAMLEOF'
socket:
  api_url: https://api.socket.dev
  fail_open: true
  fail_open_unscanned: true
  log_level: info
  api_ssl_verify: true

path_routing:
  enabled: true
  domain: localhost
  routes:
    - path: /pypi
      upstream: https://pypi.org
      registry: pypi
    - path: /npm
      upstream: https://registry.npmjs.org
      registry: npm

ports:
  http: 8080
  https: 8443

ssl:
  cert: /etc/nginx/ssl/fullchain.pem
  key: /etc/nginx/ssl/privkey.pem
YAMLEOF

  if [[ "$(docker inspect -f '{{.State.Running}}' "$REGISTRY_CONTAINER" 2>/dev/null)" != "true" ]]; then
    docker rm -f "$REGISTRY_CONTAINER" >/dev/null 2>&1 || true
    echo "Starting local registry-mode container ($REGISTRY_CONTAINER, port $REGISTRY_HTTPS_PORT)..."
    docker run -d --name "$REGISTRY_CONTAINER" \
      -p "$REGISTRY_HTTPS_PORT:8443" \
      -e SOCKET_SECURITY_API_TOKEN="$SOCKET_API_KEY" \
      -v "$REGISTRY_HOME/socket.yml:/app/socket.yml:ro" \
      -v "$REGISTRY_HOME/ssl:/etc/nginx/ssl" \
      socketdev/socket-registry-firewall:latest >/dev/null
    local waited=0
    until curl -fsk "https://localhost:$REGISTRY_HTTPS_PORT/health" >/dev/null 2>&1; do
      sleep 1; waited=$((waited + 1))
      if [[ $waited -ge 30 ]]; then
        echo "Registry-mode container did not become healthy within 30s — check:" >&2
        echo "  docker logs $REGISTRY_CONTAINER" >&2
        exit 1
      fi
    done
  fi
  echo "Registry-mode container ready: https://localhost:$REGISTRY_HTTPS_PORT"
  echo
}

# Finds the exact download URL for a PyPI package+version via the proxied simple index
# (PyPI filenames can use - or _ for the same package name, so match loosely).
pypi_download_url() {
  local pkg="$1" version="$2" name_pat href path
  name_pat="$(echo "$pkg" | sed 's/[-_.]/[-_.]/g')"
  href="$(curl -sk "https://localhost:$REGISTRY_HTTPS_PORT/pypi/simple/$pkg/" 2>/dev/null \
    | grep -oE "href=\"[^\"]*${name_pat}-${version}[^\"]*\"" | head -1 \
    | sed -E 's/^href="//; s/"$//')"
  [[ -z "$href" ]] && return
  # The index rewrites hrefs to https://<host> without our custom port (nginx assumes
  # the default HTTPS port) — strip scheme+host and re-point at our own base+port.
  path="$(echo "$href" | sed -E 's#^https?://[^/]+##')"
  echo "https://localhost:$REGISTRY_HTTPS_PORT${path}"
}

run_registry() {
  ensure_local_registry
  echo "Sending varied traffic through the local registry-mode container. Each request is a"
  echo "real GET through registry-mode's own nginx/Lua policy check — check docker logs"
  echo "$REGISTRY_CONTAINER for the [SOCKET_DECISION] line, or the Socket dashboard Events page."
  echo

  for entry in "${REGISTRY_PATTERNS[@]}"; do
    IFS='|' read -r ecosystem pkg version route demonstrates expected <<< "$entry"
    echo "--- $pkg@$version ($ecosystem) ---"
    echo "demonstrates: $demonstrates"
    echo "expected action (verified live via this deployment): $expected"
    local url status
    if [[ "$route" == "npm" ]]; then
      url="https://localhost:$REGISTRY_HTTPS_PORT/npm/$pkg/-/$pkg-$version.tgz"
    else
      url="$(pypi_download_url "$pkg" "$version")"
      if [[ -z "$url" ]]; then
        echo "Could not resolve a download URL for $pkg==$version — skipping." >&2
        echo
        continue
      fi
    fi
    set +e
    curl -sk -o /dev/null -w "download HTTP %{http_code}\n" "$url" --max-time 20
    status=$?
    set -e
    [[ $status -ne 0 ]] && echo "curl failed (exit $status) for $url" >&2
    echo
  done
  echo "Stop the container when done: docker rm -f $REGISTRY_CONTAINER"
}

case "$MODE" in
  registry) run_registry ;;
  manifest) run_manifest_batch ;;
  *) run_wrapper ;;
esac

echo "Done. In the Socket dashboard, open Events and pick a time window that covers now:"
echo "  https://socket.dev/dashboard/org/<your-org>/events?tp=1h"
echo "(verified accepted tp= values: 5m, 15m, 30m, 1h, 6h, 24h, 7d, 30d — no native"
echo "'1 day'/'2 days' option; 24h or 7d is the closest fit for that range.)"
