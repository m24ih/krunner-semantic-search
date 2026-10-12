"""KRunner D-Bus daemon for semantic file search.

Implements the org.kde.krunner1 D-Bus service interface to provide semantic
document search directly in KDE Plasma KRunner.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import mimetypes
import os
from pathlib import Path
import shutil
import signal
import subprocess
from typing import Any, Optional

from dbus_next import Variant
from dbus_next.aio import MessageBus
from dbus_next.constants import NameFlag, RequestNameReply
from dbus_next.service import ServiceInterface, method

from krunner_semantic.config import Config, load_config
from krunner_semantic.storage import StorageManager

logger = logging.getLogger("krunner_semantic.daemon")

DBUS_SERVICE_NAME = "org.kde.krunner_semantic"
DBUS_OBJECT_PATH = "/org/kde/krunner_semantic"
DBUS_INTERFACE_NAME = "org.kde.krunner1"

# Mapping common file extensions to KDE / freedesktop icon names
EXTENSION_ICONS: dict[str, str] = {
    ".pdf": "application-pdf",
    ".md": "text-markdown",
    ".markdown": "text-markdown",
    ".txt": "text-plain",
    ".text": "text-plain",
    ".py": "text-x-python",
    ".rs": "text-x-rust",
    ".c": "text-x-csrc",
    ".h": "text-x-chdr",
    ".cpp": "text-x-c++src",
    ".hpp": "text-x-c++hdr",
    ".js": "application-javascript",
    ".ts": "text-x-typescript",
    ".html": "text-html",
    ".htm": "text-html",
    ".css": "text-css",
    ".json": "application-json",
    ".yaml": "application-x-yaml",
    ".yml": "application-x-yaml",
    ".toml": "text-x-generic",
    ".csv": "text-csv",
    ".tsv": "text-plain",
    ".png": "image-x-generic",
    ".jpg": "image-x-generic",
    ".jpeg": "image-x-generic",
    ".webp": "image-x-generic",
    ".svg": "image-x-generic",
    ".gif": "image-x-generic",
    ".mp3": "audio-x-generic",
    ".wav": "audio-x-generic",
    ".flac": "audio-x-generic",
    ".ogg": "audio-x-generic",
    ".mp4": "video-x-generic",
    ".mkv": "video-x-generic",
    ".avi": "video-x-generic",
    ".webm": "video-x-generic",
    ".zip": "application-x-archive",
    ".tar": "application-x-archive",
    ".gz": "application-x-archive",
    ".xz": "application-x-archive",
    ".7z": "application-x-archive",
}


def get_file_icon(file_path: str | Path) -> str:
    """Determine the most appropriate Freedesktop icon name for a file path.

    Args:
        file_path: Path to the target file.

    Returns:
        Icon name string suitable for KDE Plasma / KRunner.
    """
    ext = Path(file_path).suffix.lower()
    if ext in EXTENSION_ICONS:
        return EXTENSION_ICONS[ext]

    guessed_mime, _ = mimetypes.guess_type(str(file_path))
    if guessed_mime:
        mime_icon = guessed_mime.replace("/", "-")
        return mime_icon

    return "text-x-generic"


class SemanticRunner(ServiceInterface):
    """Implementation of the KDE KRunner D-Bus interface (`org.kde.krunner1`).

    Allows KDE KRunner to discover actions, execute queries against the vector
    index, and open matched documents.
    """

    def __init__(
        self,
        config: Config,
        storage: StorageManager,
        embedder: Optional[Any] = None,
    ) -> None:
        """Initialize the semantic runner service.

        Args:
            config: Configuration settings.
            storage: StorageManager instance for vector searches.
            embedder: Optional embedder engine instance (e.g. EmbeddingGemma 2).
        """
        super().__init__(DBUS_INTERFACE_NAME)
        self.config = config
        self.storage = storage
        self.embedder = embedder
        self._activation_token: Optional[str] = None

    def _ensure_embedder(self) -> Optional[Any]:
        """Lazy-load the embedder if not already loaded."""
        if self.embedder is not None:
            return self.embedder

        try:
            try:
                from krunner_semantic.embedder import create_embedder

                logger.info(
                    "Initializing Embedder via create_embedder with model: %s on %s",
                    self.config.model_path,
                    self.config.effective_device,
                )
                self.embedder = create_embedder(
                    model_path=self.config.model_path,
                    device=self.config.effective_device,
                    fallback_to_mock=True,
                )
            except (ImportError, AttributeError):
                from krunner_semantic.embedder import Embedder  # type: ignore[attr-defined]

                logger.info(
                    "Initializing Embedder with model: %s on %s",
                    self.config.model_path,
                    self.config.effective_device,
                )
                self.embedder = Embedder(
                    model_path=self.config.model_path,
                    device=self.config.effective_device,
                )
            return self.embedder
        except Exception as exc:
            logger.debug("Could not initialize Embedder: %s", exc)
            return None

    def _embed_query(self, text: str) -> Optional[list[float]]:
        """Generate a 768-dimensional embedding vector for the search query."""
        engine = self._ensure_embedder()
        if engine is None:
            return None

        try:
            if hasattr(engine, "embed_query"):
                return engine.embed_query(text)
            if hasattr(engine, "embed_text"):
                return engine.embed_text(text)
            if hasattr(engine, "embed"):
                return engine.embed(text)
            if callable(engine):
                return engine(text)
        except Exception as exc:
            logger.error("Failed to compute query embedding: %s", exc)

        return None

    def actions(self) -> list[list[str]]:
        """Return secondary actions available for matched search results."""
        return [
            ["open", "Open File", "document-open"],
            ["open_folder", "Open Containing Folder", "folder-open"],
            ["copy_path", "Copy File Path", "edit-copy"],
        ]

    @method()
    def Actions(self) -> "a(sss)":
        """D-Bus method: Return secondary actions."""
        return self.actions()

    @method()
    def Config(self) -> "a{sv}":
        """Return runtime configuration for KRunner (DBus2 API support)."""
        return {
            "MinLetterCount": Variant("i", 2),
            "TriggerWords": Variant("as", ["sem:"]),
            "MatchRegex": Variant("s", r"^(sem:\s*.*|.+)$"),
        }

    def set_activation_token(self, token: str) -> None:
        """Store the XDG Activation Token for focus stealing prevention."""
        self._activation_token = token

    @method()
    def SetActivationToken(self, token: "s"):
        """D-Bus method: Store XDG Activation Token."""
        self.set_activation_token(token)

    def teardown(self) -> None:
        """Clean up state when the KRunner search session concludes."""
        self._activation_token = None

    @method()
    def Teardown(self):
        """D-Bus method: Teardown match session."""
        self.teardown()

    def match(self, query: str) -> list[list[Any]]:
        """Search semantic index and return formatted KRunner match list."""
        raw_query = query.strip()
        if not raw_query:
            return []

        has_prefix = False
        if raw_query.lower().startswith("sem:"):
            clean_query = raw_query[4:].strip()
            has_prefix = True
        else:
            clean_query = raw_query

        if len(clean_query) < 2:
            return []

        logger.debug("Received match query: '%s' (cleaned: '%s')", raw_query, clean_query)

        # Quick access to graphical settings dialog
        if clean_query.lower() in ("config", "configure", "settings", "ayar", "ayarlar", "setup"):
            return [
                [
                    "action_open_settings",
                    "Semantic Search Settings (Semantik Arama Ayarları)",
                    "preferences-system-search",
                    100,  # ExactMatch
                    1.0,
                    {
                        "subtext": Variant(
                            "s",
                            "İndekslenen klasörleri, GPU/CPU donanımını ve AI modelini yapılandırın.",
                        ),
                        "category": Variant("s", "Semantic Search Settings"),
                    },
                ]
            ]

        query_vec = self._embed_query(clean_query)
        if query_vec is None:
            if has_prefix:
                return [
                    [
                        "status_notice",
                        "Semantic Search: Model not ready",
                        "dialog-information",
                        20,  # InformationalMatch
                        0.5,
                        {
                            "subtext": Variant(
                                "s",
                                f"Model '{self.config.model_path.name}' is initializing or not indexed yet.",
                            ),
                            "category": Variant("s", "Semantic Search"),
                        },
                    ]
                ]
            return []

        try:
            # When running with MockEmbedder (before GGUF is downloaded), lower threshold
            # to 0.15 since token-averaged hashes produce scores between ~0.15 and 0.45
            effective_threshold = self.config.similarity_threshold
            embedder_instance = self._ensure_embedder()
            if embedder_instance and (
                getattr(embedder_instance, "is_mock", False)
                or embedder_instance.__class__.__name__ == "MockEmbedder"
            ):
                effective_threshold = min(effective_threshold, 0.10)

            results = self.storage.search(
                query_vector=query_vec,
                limit=self.config.max_results,
                threshold=effective_threshold,
                deduplicate_files=True,
                query_text=clean_query,
            )
            logger.debug(
                "Vector search for '%s' returned %d matches (threshold=%.2f)",
                clean_query,
                len(results),
                effective_threshold,
            )
        except Exception as exc:
            logger.error("Error executing vector search in storage: %s", exc)
            return []

        matches: list[list[Any]] = []
        for item in results:
            file_path = item["file_path"]
            path_obj = Path(file_path)

            score = float(item.get("score", 0.0))
            chunk_text = item.get("text", "")
            snippet = " ".join(chunk_text.split())[:120]
            if len(chunk_text) > 120:
                snippet += "..."

            match_id = f"{file_path}#{item.get('chunk_id', 0)}"
            file_name = path_obj.name
            icon_name = get_file_icon(file_path)

            match_type = 100 if score >= 0.75 else 50
            percentage = int(round(score * 100))
            subtext = f"{snippet} ({percentage}% match) — {path_obj.parent}"

            properties: dict[str, Variant] = {
                "subtext": Variant("s", subtext),
                "category": Variant("s", "Semantic Search"),
                "urls": Variant("as", [path_obj.as_uri()]),
                "actions": Variant("as", ["open", "open_folder", "copy_path"]),
            }

            matches.append([
                match_id,
                file_name,
                icon_name,
                match_type,
                score,
                properties,
            ])

        return matches

    @method()
    def Match(self, query: "s") -> "a(sssida{sv})":
        """D-Bus method: Search the semantic index."""
        return self.match(query)

    def run(self, match_id: str, action_id: str) -> None:
        """Execute user-selected action for a search match."""
        logger.info("Executing run for match_id='%s', action_id='%s'", match_id, action_id)

        if match_id == "status_notice":
            return

        if match_id == "action_open_settings":
            gui_bin = Path(sys.executable).parent / "krunner-semantic-config"
            if gui_bin.exists():
                subprocess.Popen([str(gui_bin)])
            else:
                subprocess.Popen([sys.executable, "-m", "krunner_semantic.gui"])
            return

        file_path_str = match_id.split("#", 1)[0]
        path_obj = Path(file_path_str).expanduser().resolve()

        env = os.environ.copy()
        if self._activation_token:
            env["XDG_ACTIVATION_TOKEN"] = self._activation_token

        try:
            if action_id == "open_folder":
                folder = path_obj.parent if path_obj.is_file() else path_obj
                subprocess.Popen(["xdg-open", str(folder)], env=env)
            elif action_id == "copy_path":
                self._copy_to_clipboard(str(path_obj))
            else:
                subprocess.Popen(["xdg-open", str(path_obj)], env=env)
        except Exception as exc:
            logger.error("Failed to run action '%s' for '%s': %s", action_id, path_obj, exc)

    @method()
    def Run(self, match_id: "s", action_id: "s"):
        """D-Bus method: Execute selected action."""
        self.run(match_id, action_id)

    def _copy_to_clipboard(self, text: str) -> None:
        """Copy text to system clipboard via wl-copy or xclip."""
        for cmd in (["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "-b", "-i"]):
            if shutil.which(cmd[0]):
                try:
                    subprocess.run(
                        cmd,
                        input=text.encode("utf-8"),
                        check=True,
                        timeout=2.0,
                    )
                    logger.debug("Copied to clipboard using %s", cmd[0])
                    return
                except Exception as exc:
                    logger.debug("Clipboard command %s failed: %s", cmd[0], exc)


async def run_daemon(config: Config) -> None:
    """Run the KRunner Semantic Search D-Bus service loop.

    Args:
        config: Configuration settings.
    """
    logger.info("Connecting to session D-Bus...")
    bus = await MessageBus().connect()

    logger.info("Opening database at %s...", config.db_path)
    storage = StorageManager(config.db_path)

    runner = SemanticRunner(config=config, storage=storage)
    bus.export(DBUS_OBJECT_PATH, runner)

    reply = await bus.request_name(
        DBUS_SERVICE_NAME,
        flags=NameFlag.REPLACE_EXISTING | NameFlag.ALLOW_REPLACEMENT,
    )

    if reply not in (RequestNameReply.PRIMARY_OWNER, RequestNameReply.ALREADY_OWNER):
        logger.error(
            "Could not become primary owner of '%s' (reply=%s). Another instance may be running.",
            DBUS_SERVICE_NAME,
            reply,
        )
        bus.disconnect()
        return

    logger.info(
        "Successfully registered '%s' on %s. Ready to serve KRunner requests.",
        DBUS_SERVICE_NAME,
        DBUS_OBJECT_PATH,
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, RuntimeError):
            pass

    try:
        await stop_event.wait()
    finally:
        logger.info("Shutting down KRunner Semantic Search daemon...")
        storage.close()
        bus.disconnect()


def main() -> None:
    """CLI entrypoint for krunner-semantic-daemon."""
    parser = argparse.ArgumentParser(
        description="KRunner Semantic Search D-Bus Daemon (EmbeddingGemma 2 + SQLite-Vec)"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Path to custom config.toml file",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default=None,
        help="Override compute device",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging output",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    if args.device:
        config.device = args.device

    try:
        asyncio.run(run_daemon(config))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
