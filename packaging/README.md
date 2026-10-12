# Packaging Guide for KRunner Semantic Search

This directory contains packaging specifications and build automation recipes for **KRunner Semantic Search** across major Linux distribution ecosystems:

- **Arch Linux (AUR)**: [PKGBUILD](file:///home/melih/Projects/krunner-semantic-search/packaging/arch/PKGBUILD)
- **Fedora (COPR)**: [krunner-semantic-search.spec](file:///home/melih/Projects/krunner-semantic-search/packaging/fedora/krunner-semantic-search.spec) and [.copr/Makefile](file:///home/melih/Projects/krunner-semantic-search/.copr/Makefile)
- **Terra / Ultramarine Linux**: [anda.hcl](file:///home/melih/Projects/krunner-semantic-search/packaging/terra/anda.hcl) and [krunner-semantic-search.spec](file:///home/melih/Projects/krunner-semantic-search/packaging/terra/krunner-semantic-search.spec)

---

## Table of Contents
1. [Installed System Files](#installed-system-files)
2. [Arch Linux (AUR)](#arch-linux-aur)
   - [Local Build & Testing](#arch-local-build)
   - [Publishing to the AUR](#arch-publishing-to-the-aur)
   - [End-User Installation](#arch-end-user-installation)
3. [Fedora (COPR)](#fedora-copr)
   - [Spec File & Automation](#fedora-spec-file--automation)
   - [Local Build (SRPM & RPM)](#fedora-local-build)
   - [Setting up Fedora COPR Repository](#fedora-setting-up-copr)
   - [End-User Installation](#fedora-end-user-installation)
4. [Terra & Ultramarine Linux (Anda)](#terra--ultramarine-linux-anda)
   - [Overview of Terra & Anda](#terra-overview)
   - [Local Build with Anda](#terra-local-build)
   - [Contributing to Terra Monorepo](#terra-contributing)
   - [End-User Installation](#terra-end-user-installation)
5. [Post-Installation & Service Setup](#post-installation--service-setup)

---

## Installed System Files

Across all package formats, the build installs the following components:

| Component | Target Location | Description |
| :--- | :--- | :--- |
| **CLI Executables** | `/usr/bin/krunner-semantic-*` | Daemon, Indexer, and Config GUI entry points |
| **Python Library** | `/usr/lib/python3.*/site-packages/krunner_semantic/` | Core semantic search logic & models |
| **KRunner Plugin** | `/usr/share/krunner/dbusplugins/plasma-runner-semantic.desktop` | D-Bus runner definition registered to KDE Plasma 6 |
| **Desktop Launcher** | `/usr/share/applications/krunner-semantic-config.desktop` | Graphical configuration launcher |
| **Systemd User Unit** | `/usr/lib/systemd/user/krunner-semantic.service` | Per-user background service management |

---

## Arch Linux (AUR)

### Arch Local Build
To test the Arch package locally:

```bash
cd packaging/arch
makepkg -si
```

This compiles the Python wheel, runs `python -m installer`, and places the runner plugin and systemd user unit into the Arch package hierarchy.

### Arch Publishing to the AUR
1. Clone the AUR repository:
   ```bash
   git clone ssh://aur@aur.archlinux.org/krunner-semantic-search-git.git
   cd krunner-semantic-search-git
   ```
2. Copy `packaging/arch/PKGBUILD` into the repository.
3. Generate the metadata `.SRCINFO` file:
   ```bash
   makepkg --printsrcinfo > .SRCINFO
   ```
4. Commit and push:
   ```bash
   git add PKGBUILD .SRCINFO
   git commit -m "feat: release version 0.1.0"
   git push origin master
   ```

### Arch End-User Installation
Users can install using any AUR helper:

```bash
# Using yay
yay -S krunner-semantic-search-git

# Using paru
paru -S krunner-semantic-search-git
```

---

## Fedora (COPR)

### Fedora Spec File & Automation
- Spec file: `packaging/fedora/krunner-semantic-search.spec`
- COPR build entry point: `.copr/Makefile`

The Fedora spec uses standard Python RPM macros (`%pyproject_wheel`, `%pyproject_install`, `%pyproject_save_files`) and systemd user unit macros (`%systemd_user_post`, `%systemd_user_preun`, `%systemd_user_postun_with_restart`).

### Fedora Local Build
To test the RPM build on a Fedora workstation or container:

1. Install build dependencies:
   ```bash
   sudo dnf install -y rpm-build python3-devel python3-pip python3-build python3-installer \
                       python3-wheel python3-hatchling systemd-rpm-macros desktop-file-utils
   ```
2. Build the Source RPM (SRPM):
   ```bash
   make -f .copr/Makefile srpm outdir=.
   ```
3. Build the binary RPM:
   ```bash
   rpmbuild --rebuild krunner-semantic-search-0.1.0-1.fc*.src.rpm
   ```

### Fedora Setting up COPR
1. Log in to [Fedora COPR](https://copr.fedorainfracloud.org/).
2. Create a new project: `krunner-semantic-search`.
3. Add a new package build source:
   - **Type**: `Git / Custom script`
   - **Clone URL**: `https://github.com/m24ih/krunner-semantic-search.git`
   - **Subdirectory**: Leave root (`.`)
   - **Spec file path**: `packaging/fedora/krunner-semantic-search.spec`
   - COPR will automatically detect `.copr/Makefile` and invoke the `srpm` target.
4. Trigger a build. COPR will generate packages for Fedora 40, Fedora 41, and Rawhide.

### Fedora End-User Installation
Once built in COPR, users install via:

```bash
sudo dnf copr enable m24ih/krunner-semantic-search
sudo dnf install krunner-semantic-search
```

---

## Terra & Ultramarine Linux (Anda)

### Terra Overview
[Terra](https://github.com/terrapkg/packages) is Fyra Labs' flagship package repository, standard on **Ultramarine Linux** and available for Fedora. Terra utilizes **Andaman (`anda`)** as its meta-build tool.

- Anda manifest: `packaging/terra/anda.hcl`
- RPM spec: `packaging/terra/krunner-semantic-search.spec`

### Terra Local Build
If `anda` and `mock` are installed:

```bash
cd packaging/terra
anda build
```

This validates the spec against mock chroots configured by Andaman.

### Terra Contributing
To submit KRunner Semantic Search to Terra:
1. Fork the `terrapkg/packages` repository.
2. Create a folder `krunner-semantic-search/` under the appropriate category.
3. Copy `packaging/terra/anda.hcl` and `packaging/terra/krunner-semantic-search.spec`.
4. Open a pull request against the `terrapkg/packages` repository. Once merged, it is automatically built and distributed to all Terra users.

### Terra End-User Installation
On Ultramarine Linux (where Terra is pre-configured):

```bash
sudo dnf install krunner-semantic-search
```

On Fedora with Terra enabled:

```bash
sudo dnf install --repofrompath=terra,https://repos.fyralabs.com/terra$releasever \
                 --setopt=terra.gpgkey=https://repos.fyralabs.com/terra$releasever/RPM-GPG-KEY-terra \
                 krunner-semantic-search
```

---

## Post-Installation & Service Setup

After installing the package on any distribution, enable and start the systemd user service:

```bash
# 1. Enable and start the background daemon
systemctl --user enable --now krunner-semantic.service

# 2. Check service status
systemctl --user status krunner-semantic.service

# 3. Reload KRunner to ensure plugin discovery
kquitapp6 krunner 2>/dev/null || true
krunner & disown 2>/dev/null || true
```

### Initial Model Download & Usage
1. Open the settings dialog:
   ```bash
   krunner-semantic-config
   ```
   Or in KRunner (`Alt + Space`), type:
   ```text
   sem: settings
   ```
2. Download the official **EmbeddingGemma 2** model.
3. In KRunner, search your documents with:
   ```text
   sem: <query>
   ```
