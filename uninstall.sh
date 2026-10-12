#!/usr/bin/env bash
# ==============================================================================
# KRunner Semantic Search - Uninstaller
# ==============================================================================
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Stopping background service...${NC}"
systemctl --user stop krunner-semantic.service 2>/dev/null || true
systemctl --user disable krunner-semantic.service 2>/dev/null || true

echo -e "${YELLOW}Removing files...${NC}"
rm -f "${HOME}/.local/share/krunner/dbusplugins/plasma-runner-semantic.desktop"
rm -f "${HOME}/.local/share/applications/krunner-semantic-config.desktop"
rm -f "${HOME}/.config/systemd/user/krunner-semantic.service"
rm -f "${HOME}/.local/bin/krunner-semantic-daemon"
rm -f "${HOME}/.local/bin/krunner-semantic-indexer"
rm -f "${HOME}/.local/bin/krunner-semantic-config"

read -p "Do you also want to remove the database and model files (~/.local/share/krunner-semantic)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "${HOME}/.local/share/krunner-semantic"
    rm -rf "${HOME}/.config/krunner-semantic"
    echo -e "${GREEN}Data directories removed.${NC}"
fi

systemctl --user daemon-reload
kquitapp6 krunner 2>/dev/null || true
krunner & disown 2>/dev/null || true

echo -e "${GREEN}KRunner Semantic Search has been uninstalled.${NC}"
