#!/usr/bin/env bash
# ==============================================================================
# KRunner Semantic Search - Universal Linux Installer
# 100% Local, Privacy-First Semantic File Search for KDE Plasma 6
# ==============================================================================
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}====================================================================${NC}"
echo -e "${BLUE}          KRunner Semantic Search - Installer                      ${NC}"
echo -e "${BLUE}====================================================================${NC}"

# Detect Directories
INSTALL_DIR="${HOME}/.local/share/krunner-semantic"
VENV_DIR="${INSTALL_DIR}/venv"
BIN_DIR="${HOME}/.local/bin"
DBUS_PLUGINS_DIR="${HOME}/.local/share/krunner/dbusplugins"
APPLICATIONS_DIR="${HOME}/.local/share/applications"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${INSTALL_DIR}/models" "${BIN_DIR}" "${DBUS_PLUGINS_DIR}" "${APPLICATIONS_DIR}" "${SYSTEMD_USER_DIR}"

# 1. Check Python 3
echo -e "\n${YELLOW}[1/6] Checking Python runtime...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 is required but not installed.${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "Found Python ${GREEN}${PYTHON_VERSION}${NC}"

# 2. Setup Virtual Environment
echo -e "\n${YELLOW}[2/6] Setting up virtual environment in ${VENV_DIR}...${NC}"
if command -v uv &> /dev/null; then
    echo "Using fast package installer 'uv'..."
    uv venv "${VENV_DIR}" --python python3
    uv pip install --python "${VENV_DIR}/bin/python" -e "${REPO_DIR}"
else
    echo "Using standard python venv and pip..."
    python3 -m venv "${VENV_DIR}"
    "${VENV_DIR}/bin/pip" install --upgrade pip
    "${VENV_DIR}/bin/pip" install -e "${REPO_DIR}"
fi

# 3. Create CLI Symlinks in ~/.local/bin
echo -e "\n${YELLOW}[3/6] Linking executables to ${BIN_DIR}...${NC}"
for script in krunner-semantic-daemon krunner-semantic-indexer krunner-semantic-config; do
    ln -sf "${VENV_DIR}/bin/${script}" "${BIN_DIR}/${script}"
    echo -e "  Linked ${GREEN}${BIN_DIR}/${script}${NC}"
done

# 4. Install Desktop Files & KRunner D-Bus Runner
echo -e "\n${YELLOW}[4/6] Registering KRunner D-Bus Runner and Settings Launcher...${NC}"
# KRunner runner registration
cp "${REPO_DIR}/desktop/plasma-runner-semantic.desktop" "${DBUS_PLUGINS_DIR}/plasma-runner-semantic.desktop"
echo -e "  Installed ${GREEN}${DBUS_PLUGINS_DIR}/plasma-runner-semantic.desktop${NC}"

# Application launcher
sed "s|/home/melih/Projects/krunner-semantic-search/.venv/bin/krunner-semantic-config|${BIN_DIR}/krunner-semantic-config|g" \
    "${REPO_DIR}/desktop/krunner-semantic-config.desktop" > "${APPLICATIONS_DIR}/krunner-semantic-config.desktop"
chmod +x "${APPLICATIONS_DIR}/krunner-semantic-config.desktop"
echo -e "  Installed ${GREEN}${APPLICATIONS_DIR}/krunner-semantic-config.desktop${NC}"

# 5. Setup Systemd User Service
echo -e "\n${YELLOW}[5/6] Configuring systemd background daemon...${NC}"
cp "${REPO_DIR}/systemd/user/krunner-semantic.service" "${SYSTEMD_USER_DIR}/krunner-semantic.service"
systemctl --user daemon-reload
systemctl --user enable --now krunner-semantic.service
echo -e "  Enabled and started ${GREEN}krunner-semantic.service${NC}"

# 6. Refresh KRunner
echo -e "\n${YELLOW}[6/6] Reloading KRunner...${NC}"
kquitapp6 krunner 2>/dev/null || true
krunner & disown 2>/dev/null || true

echo -e "\n${GREEN}====================================================================${NC}"
echo -e "${GREEN}✓ Installation complete successfully!                               ${NC}"
echo -e "${GREEN}====================================================================${NC}"
echo -e "You can now:"
echo -e "  1. Open KRunner (${YELLOW}Alt + Space${NC}) and type: ${YELLOW}sem: <your query>${NC}"
echo -e "  2. Open GUI Settings by typing ${YELLOW}sem: settings${NC} in KRunner, or running:"
echo -e "     ${YELLOW}krunner-semantic-config${NC}"
echo -e "  3. Download the official EmbeddingGemma model directly from the Settings GUI."
