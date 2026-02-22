#!/usr/bin/env bash
# Install Dispatcher as a macOS LaunchAgent.
# This is a convenience wrapper around `dispatcher install`.
#
# Usage:
#   ./scripts/install-macos.sh          # Install and start
#   ./scripts/install-macos.sh remove   # Uninstall

set -euo pipefail

LABEL="com.dispatcher.agent"
PLIST_DIR="$HOME/Library/LaunchAgents"
PLIST="$PLIST_DIR/$LABEL.plist"

if [[ "${1:-}" == "remove" ]]; then
    echo "Uninstalling Dispatcher..."
    launchctl unload "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    echo "Done."
    exit 0
fi

# Check prerequisites
if ! command -v dispatcher &>/dev/null; then
    echo "Error: 'dispatcher' not found. Install with: pip install agent-dispatcher"
    exit 1
fi

if ! command -v claude &>/dev/null; then
    echo "Warning: 'claude' not found in PATH. Make sure your agent command is installed."
fi

CONFIG="$HOME/.config/dispatcher/config.yaml"
if [[ ! -f "$CONFIG" ]]; then
    echo "No config found. Running interactive setup..."
    dispatcher init
fi

echo "Installing Dispatcher as LaunchAgent..."
dispatcher install

echo ""
echo "Dispatcher is now running and will auto-start on login."
echo ""
echo "Useful commands:"
echo "  dispatcher status      Check if running"
echo "  dispatcher logs -f     Follow logs"
echo "  dispatcher uninstall   Remove LaunchAgent"
