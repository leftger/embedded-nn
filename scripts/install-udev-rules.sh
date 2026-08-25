#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RULES_SRC="$SCRIPT_DIR/../udev/99-embedded-nn.rules"
RULES_DEST="/etc/udev/rules.d/99-embedded-nn.rules"

echo "Installing embedded-nn udev rules to $RULES_DEST..."
sudo cp "$RULES_SRC" "$RULES_DEST"
sudo udevadm control --reload-rules
sudo udevadm trigger
echo "Done! Please replug your STM32WBA65 / ST-Link device."
