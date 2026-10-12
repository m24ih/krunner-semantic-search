# KRunner Semantic Search (EmbeddingGemma 2)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![KDE Plasma](https://img.shields.io/badge/KDE%20Plasma-6.x-3183cf.svg)](https://kde.org/plasma-desktop/)
[![Model](https://img.shields.io/badge/Model-EmbeddingGemma%202%20(GGUF)-orange.svg)](https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)

**Fast, 100% local, privacy-first semantic file search plugin and graphical manager for KDE Plasma 6 KRunner powered by Google DeepMind's EmbeddingGemma 2 model.**

Unlike traditional keyword-based desktop search engines (e.g. Baloo) that strictly match exact filenames or raw substrings, **KRunner Semantic Search** understands the contextual meaning of your files. Search naturally using descriptions, concepts, or partial ideas (e.g., `"rental agreement pdf"`, `"vector database benchmark"`, `"machine learning syllabus"`) even when you cannot remember the exact file name.

---

## ✨ Features

- **100% Local & Private:** All embeddings and vector calculations run strictly on your machine. No telemetry, no cloud APIs, and zero network leakage.
- **Hybrid Search Engine:** Combines instant lexical filename matching with dense vector KNN cosine distance (`sqlite-vec`). Searching for exact or partial file names hits immediately with 95% relevance, while conceptual queries retrieve contextual document chunks.
- **Hardware-Aware Acceleration:** Automatically detects and offloads inference to NVIDIA CUDA (`RTX 4060` tested at ~10ms per query) with seamless graceful degradation to multi-threaded CPU mode on systems without a dedicated GPU.
- **Native Qt6 Graphical Manager (`krunner-semantic-config`):** Modern desktop GUI to manage watched folders, view real-time indexing progress (%xx), switch compute devices, and adjust search thresholds.
- **One-Click Model Installer:** Download and configure the official 333 MB EmbeddingGemma 2 GGUF model directly from the settings interface.
- **Lightweight D-Bus Daemon:** Runs asynchronously via `org.kde.krunner1`, ensuring zero typing latency or UI lag inside KRunner.
- **Comprehensive Document Parsers:** Out-of-the-box support for plain text, Markdown, codebases (`.py`, `.rs`, `.cpp`, `.json`, etc.), PDF documents (`pypdfium2`), and image metadata (`Pillow`).

---

## 🏗️ Architecture

```
[ KDE KRunner Interface ]
           │  (Session D-Bus: org.kde.krunner1)
           ▼
┌────────────────────────────────────────────────────────┐
│  krunner-semantic-daemon (D-Bus Service)               │
│  - Receives live query as you type                     │
│  - Generates 768-dim normalized embedding vector       │
│  - Performs hybrid query (Lexical + SQLite-Vec KNN)    │
│  - Returns ranked results with desktop MIME icons      │
└────────────────────────────────────────────────────────┘
           ▲
           │ (Read)
┌──────────┴─────────────────────────────────────────────┐
│  ~/.local/share/krunner-semantic/index.db              │
│  (Embedded SQLite-Vec Virtual Table)                   │
└────────────────────────────────────────────────────────┘
           ▲
           │ (Write / Incremental Sync)
┌──────────┴─────────────────────────────────────────────┐
│  krunner-semantic-indexer & GUI Manager                │
│  - Recursively watches folders (~/Documents, etc.)     │
│  - Detects mtime/size changes with hash verification   │
│  - Splits text into sliding window chunks              │
│  - Calculates vectors via EmbeddingGemma 2 (CUDA/CPU)  │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Installation

### Option 1: Arch Linux (AUR)
```bash
yay -S krunner-semantic-search-git
# or with paru:
paru -S krunner-semantic-search-git
```

### Option 2: Fedora (COPR)
```bash
sudo dnf copr enable m24ih/krunner-semantic-search
sudo dnf install krunner-semantic-search
```

### Option 3: Terra / Ultramarine Linux
```bash
# Ultramarine Linux or Fedora with Terra repository enabled:
sudo dnf install krunner-semantic-search
```

### Option 4: Universal One-Liner (Any Linux Distribution)
```bash
curl -sSL https://raw.githubusercontent.com/m24ih/krunner-semantic-search/main/install.sh | bash
```

### Option 5: Manual Installation (From Source)
```bash
git clone https://github.com/m24ih/krunner-semantic-search.git
cd krunner-semantic-search
./install.sh
```

The installer automatically:
1. Prepares an isolated Python virtual environment with all dependencies.
2. Symlinks `krunner-semantic-daemon`, `krunner-semantic-indexer`, and `krunner-semantic-config` to `~/.local/bin/`.
3. Registers the KRunner D-Bus plugin in `~/.local/share/krunner/dbusplugins/`.
4. Installs the desktop launcher in `~/.local/share/applications/`.
5. Enables and starts the background `systemd --user` service.

---

## 💡 Usage

### 1. Searching in KRunner
Press **Alt + Space** (or **Alt + F2**) to trigger KRunner:
```text
sem: <your natural language query>
```
*Examples:*
- `sem: machine learning transcript`
- `sem: philips coffee machine descaling`
- `sem: yamtrack`
- `sem: invoice 2026`

Pressing **Enter** opens the file in your default desktop application, or use the action menu to open the containing folder or copy the file path.

### 2. Opening the Settings GUI
You can open the configuration interface in three ways:
1. **Via KRunner:** Type `sem: settings` or `sem: config` and hit Enter.
2. **Via Application Menu:** Search for **"Semantic Search Settings"**.
3. **Via Terminal:**
   ```bash
   krunner-semantic-config
   ```

Inside the settings window, you can:
- Add or remove indexed folders.
- Click **"Download EmbeddingGemma 2 (333 MB)"** for one-click model setup.
- Watch real-time indexing progress percentage (`xx% — Indexing: report.pdf [120/267]`).
- Toggle between **Auto (CUDA GPU)** and **CPU** inference modes.

### 3. Command Line Interface (CLI)

```bash
# Scan and index all configured watch directories incrementally:
krunner-semantic-indexer --scan

# Force re-index of all files regardless of modification time:
krunner-semantic-indexer --scan --force

# Inspect current index database statistics:
krunner-semantic-indexer --status

# Index a single file or custom directory:
krunner-semantic-indexer --file ~/Downloads/document.pdf
krunner-semantic-indexer --directory ~/Projects
```

---

## ⚙️ Configuration File

Configuration is saved in `~/.config/krunner-semantic/config.toml`:

```toml
[inference]
# Device target: 'auto' (preferred), 'cuda', or 'cpu'
device = "auto"
# Path to EmbeddingGemma GGUF weights
model_path = "~/.local/share/krunner-semantic/models/embeddinggemma-300M-Q8_0.gguf"

[storage]
db_path = "~/.local/share/krunner-semantic/index.db"

[indexer]
watch_directories = [
    "~/Documents",
    "~/Notes"
]

[search]
max_results = 5
similarity_threshold = 0.55
```

---

## 🗑️ Uninstallation

To cleanly remove the application, service, and plugin:
```bash
./uninstall.sh
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
