#!/usr/bin/env bash
# Publish all Cortex packages to PyPI.
# Usage: UV_PUBLISH_TOKEN=pypi-xxx ./scripts/publish.sh
# Or:    ./scripts/publish.sh <package-name>  (publish single package)
set -euo pipefail

PACKAGES=(vibe-replay agent-dispatcher forge-agent a2a-hub cortex-cli-agent cortex-memory)

if [ -z "${UV_PUBLISH_TOKEN:-}" ]; then
  echo "Error: UV_PUBLISH_TOKEN is not set."
  echo "Get a token from https://pypi.org/manage/account/token/"
  exit 1
fi

if [ $# -gt 0 ]; then
  PACKAGES=("$1")
fi

for pkg in "${PACKAGES[@]}"; do
  echo "==> Building ${pkg}..."
  rm -f dist/*
  uv build --package "${pkg}"
  echo "==> Publishing ${pkg}..."
  uv publish dist/*
  echo "==> ${pkg} published."
  echo ""
done

echo "All packages published."
