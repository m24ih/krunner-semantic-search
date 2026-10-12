"""Configuration management and hardware detection for KRunner Semantic Search."""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import subprocess
from typing import Optional

# Use tomllib from standard library (Python 3.11+) or fall back to tomli
try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = Path.home() / ".config" / "krunner-semantic"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.toml"
DEFAULT_DATA_DIR = Path.home() / ".local" / "share" / "krunner-semantic"
DEFAULT_MODEL_PATH = DEFAULT_DATA_DIR / "models" / "embeddinggemma-300M-Q8_0.gguf"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "index.db"
DEFAULT_WATCH_DIRS = [Path.home() / "Documents", Path.home() / "Notes"]


def detect_device() -> str:
    """Gracefully detect whether a CUDA-compatible GPU is available.

    Checks via torch (if available), nvidia-smi execution, or libcuda via ctypes.
    Safely falls back to 'cpu' if any check fails or no GPU is present.

    Returns:
        'cuda' if an operational NVIDIA GPU with CUDA support is detected,
        otherwise 'cpu'.
    """
    # 1. Check PyTorch CUDA availability if installed
    try:
        import torch

        if torch.cuda.is_available():
            logger.debug("CUDA detected via PyTorch")
            return "cuda"
    except Exception:
        pass

    # 2. Check via nvidia-smi command-line tool
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=2.0,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                logger.debug("CUDA detected via nvidia-smi: %s", result.stdout.strip().splitlines()[0])
                return "cuda"
        except Exception as exc:
            logger.debug("nvidia-smi check failed: %s", exc)

    # 3. Check via CUDA driver library (libcuda.so) using ctypes
    try:
        libcuda_name = ctypes.util.find_library("cuda") or "libcuda.so.1"
        cuda_lib = ctypes.CDLL(libcuda_name)
        # cuInit(0) initializes the CUDA driver API
        if hasattr(cuda_lib, "cuInit") and cuda_lib.cuInit(0) == 0:
            logger.debug("CUDA detected via libcuda ctypes binding")
            return "cuda"
    except Exception as exc:
        logger.debug("libcuda ctypes check failed: %s", exc)

    return "cpu"


PRESET_EXCLUDES: dict[str, dict[str, str]] = {
    "git": {
        "label": "Git Repositories (.git)",
        "pattern": r"(^|/)\.git(/|$)",
    },
    "node_modules": {
        "label": "Node.js Dependencies (node_modules)",
        "pattern": r"(^|/)node_modules(/|$)",
    },
    "python_venv": {
        "label": "Python & UV Environments / Cache (.venv, venv, __pycache__)",
        "pattern": r"(^|/)(\.venv|venv|__pycache__)(/|$)",
    },
    "build_cache": {
        "label": "Build & Distribution Caches (build, dist, .cache, .tox)",
        "pattern": r"(^|/)(build|dist|\.cache|\.tox)(/|$)",
    },
    "ide": {
        "label": "IDE & Editor Metadata (.idea, .vscode)",
        "pattern": r"(^|/)(\.idea|\.vscode)(/|$)",
    },
}


@dataclass
class Config:
    """Configuration settings for KRunner Semantic Search.

    Attributes:
        device: Compute device ('auto', 'cuda', or 'cpu').
        model_path: Path to the EmbeddingGemma GGUF model file.
        db_path: Path to the SQLite-Vec database file.
        watch_directories: List of directories to index and monitor.
        max_results: Maximum number of search matches to return to KRunner.
        similarity_threshold: Minimum cosine similarity score (0.0 to 1.0).
        index_hidden_files: Whether to discover and index hidden dotfiles (.config, .bashrc).
        enabled_presets: List of active preset exclusion keys (e.g. 'git', 'node_modules').
        custom_excludes: List of custom pattern dicts: [{'pattern': str, 'enabled': bool}].
    """

    device: str = "auto"
    model_path: Path = field(default_factory=lambda: DEFAULT_MODEL_PATH)
    db_path: Path = field(default_factory=lambda: DEFAULT_DB_PATH)
    watch_directories: list[Path] = field(default_factory=lambda: list(DEFAULT_WATCH_DIRS))
    max_results: int = 5
    similarity_threshold: float = 0.55
    index_hidden_files: bool = False
    enabled_presets: list[str] = field(default_factory=lambda: list(PRESET_EXCLUDES.keys()))
    custom_excludes: list[dict[str, Any]] = field(default_factory=list)

    @property
    def effective_device(self) -> str:
        """Resolve the active compute device ('cuda' or 'cpu')."""
        if self.device == "auto":
            return detect_device()
        return self.device.lower()

    @property
    def active_exclude_patterns(self) -> list[str]:
        """Return combined list of all currently active regex exclusion patterns."""
        patterns: list[str] = []
        for key in self.enabled_presets:
            if key in PRESET_EXCLUDES:
                patterns.append(PRESET_EXCLUDES[key]["pattern"])
        for item in self.custom_excludes:
            if item.get("enabled", True) and item.get("pattern"):
                patterns.append(item["pattern"])
        return patterns


def _generate_default_toml(config: Config) -> str:
    """Generate clean, commented TOML content for the configuration."""
    watch_dirs_str = ",\n".join(f'    {json.dumps(str(p))}' for p in config.watch_directories)
    enabled_presets_str = ", ".join(json.dumps(k) for k in config.enabled_presets)

    custom_lines = []
    for item in config.custom_excludes:
        pat_json = json.dumps(str(item.get("pattern", "")))
        en = "true" if item.get("enabled", True) else "false"
        custom_lines.append(f'    {{ pattern = {pat_json}, enabled = {en} }}')
    custom_excludes_str = ",\n".join(custom_lines)

    return f"""# KRunner Semantic Search Configuration
# Automatically generated configuration file

# Compute device: "auto" (detects CUDA or falls back to CPU), "cuda", or "cpu"
device = {json.dumps(config.device)}

# Path to the EmbeddingGemma 2 GGUF embedding model
model_path = {json.dumps(str(config.model_path))}

# SQLite-Vec vector database path
db_path = {json.dumps(str(config.db_path))}

# Directories to index and monitor for documents
watch_directories = [
{watch_dirs_str}
]

# Maximum number of results to display in KRunner
max_results = {config.max_results}

# Minimum cosine similarity threshold (range 0.0 to 1.0)
similarity_threshold = {config.similarity_threshold}

# Whether to index hidden files and directories (starting with .)
index_hidden_files = {"true" if config.index_hidden_files else "false"}

# Predefined exclusion presets: "git", "node_modules", "python_venv", "build_cache", "ide"
enabled_presets = [{enabled_presets_str}]

# Custom regex exclusion patterns with toggleable state
custom_excludes = [
{custom_excludes_str}
]
"""


def save_config(config: Config, path: Optional[Path] = None) -> None:
    """Save configuration to a TOML file.

    Args:
        config: The Config instance to save.
        path: Optional file path. Defaults to ~/.config/krunner-semantic/config.toml.
    """
    target_path = (path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    toml_content = _generate_default_toml(config)
    target_path.write_text(toml_content, encoding="utf-8")


def load_config(path: Optional[Path] = None) -> Config:
    """Load configuration from ~/.config/krunner-semantic/config.toml.

    If the configuration file does not exist, creates the file with default values
    and returns a default Config instance.

    Args:
        path: Optional custom config file path. Defaults to standard user config.

    Returns:
        Config instance populated with user or default settings.
    """
    config_file = (path or DEFAULT_CONFIG_PATH).expanduser()

    if not config_file.exists():
        logger.info("Config file not found at %s. Creating defaults.", config_file)
        default_cfg = Config()
        try:
            save_config(default_cfg, config_file)
        except OSError as exc:
            logger.warning("Failed to write default config file to %s: %s", config_file, exc)
        return default_cfg

    if tomllib is None:
        logger.warning("No TOML parser available (install tomli or use Python 3.11+). Using defaults.")
        return Config()

    try:
        with config_file.open("rb") as f:
            data = tomllib.load(f)
    except Exception as exc:
        logger.error("Error reading configuration from %s: %s. Using defaults.", config_file, exc)
        return Config()

    device = str(data.get("device", "auto"))

    raw_model_path = data.get("model_path", str(DEFAULT_MODEL_PATH))
    model_path = Path(raw_model_path).expanduser()

    raw_db_path = data.get("db_path", str(DEFAULT_DB_PATH))
    db_path = Path(raw_db_path).expanduser()

    raw_watch_dirs = data.get("watch_directories", [str(d) for d in DEFAULT_WATCH_DIRS])
    watch_directories = [Path(d).expanduser() for d in raw_watch_dirs]

    max_results = int(data.get("max_results", 5))
    similarity_threshold = float(data.get("similarity_threshold", 0.55))

    index_hidden_files = bool(data.get("index_hidden_files", False))
    enabled_presets = list(data.get("enabled_presets", list(PRESET_EXCLUDES.keys())))
    custom_excludes = list(data.get("custom_excludes", []))

    return Config(
        device=device,
        model_path=model_path,
        db_path=db_path,
        watch_directories=watch_directories,
        max_results=max_results,
        similarity_threshold=similarity_threshold,
        index_hidden_files=index_hidden_files,
        enabled_presets=enabled_presets,
        custom_excludes=custom_excludes,
    )
