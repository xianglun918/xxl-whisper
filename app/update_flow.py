"""Update flow: silent auto-check watcher plus user-triggered check/download.

Auto path never blocks or steals focus: it posts one tray notification when a
newer GitHub release appears. Manual path (tray menu) shows a version dialog.
On a frozen Windows build it then downloads the new exe, verifies its published
checksum, and hands off to the self-update helper for a one-click in-place
upgrade; everywhere else (and on any failure) it degrades to opening the
download page. Network failures become a log line (auto) or a small info box
(manual).
"""

import logging
import sys
import threading
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app import native
from app.config import updates_dir
from app.download_io import DownloadError
from app.download_io import download as download_file
from app.selfupdate import (
    build_update_spec,
    download_filename,
    launch_helper,
    supports_self_update,
)
from app.tray import Tray
from app.updater import (
    ReleaseInfo,
    UpdateCheckError,
    fetch_checksum,
    fetch_latest_release,
    find_update_assets,
    is_newer,
    parse_version,
)

log = logging.getLogger(__name__)

_INITIAL_DELAY_S: float = 15.0
_RECHECK_INTERVAL_S: float = 24 * 3600.0


@dataclass(frozen=True, slots=True)
class UpdateFlowDeps:
    """Live dependencies the update flow reports through."""

    tray: Tray
    indicator: native.indicator.Indicator
    current_version: str
    on_quit: Callable[[], None]
    get_proxy: Callable[[], str]


class UpdateFlow:
    """Owns the update-watcher thread and the manual check/download flow."""

    def __init__(self, deps: UpdateFlowDeps) -> None:
        self._deps = deps
        self._current = parse_version(deps.current_version)
        self._stop_event = threading.Event()

    def start_watcher(self, enabled: bool) -> None:
        if not enabled:
            log.info("update checks disabled by config")
            return
        threading.Thread(target=self._watch_loop, daemon=True, name="update-watcher").start()

    def stop(self) -> None:
        self._stop_event.set()

    def manual_check(self) -> None:
        """Tray-menu entry point; safe to run on any thread (shows a dialog)."""
        release = self._fetch_or_report()
        if release is None:
            return
        if not is_newer(release.version, self._current):
            current_text = ".".join(map(str, self._current))
            native.util.show_info(f"已是最新版本 {current_text}。")
            return
        if supports_self_update() and find_update_assets(release) is not None:
            if self._confirm_install(release):
                self._install(release)
            return
        if self._offer(release):
            log.info("update: opening %s", release.url)
            webbrowser.open(release.url)

    # -- internals -----------------------------------------------------------

    def _watch_loop(self) -> None:
        self._stop_event.wait(_INITIAL_DELAY_S)
        while not self._stop_event.is_set():
            release = self._fetch_or_report()
            if release is not None and is_newer(release.version, self._current):
                log.info("update available: %s", release.tag)
                self._deps.tray.notify(
                    f"发现新版本 {release.tag}：右键托盘 → 检查更新",
                    title="xxl-whisper 可升级",
                )
            self._stop_event.wait(_RECHECK_INTERVAL_S)

    def _fetch_or_report(self) -> ReleaseInfo | None:
        try:
            return fetch_latest_release()
        except UpdateCheckError as exc:
            log.info("update check skipped: %s", exc)
            return None

    def _confirm_install(self, release: ReleaseInfo) -> bool:
        current_text = ".".join(map(str, self._current))
        message = (
            f"发现新版本 {release.tag}（当前 v{current_text}）。\n\n"
            "现在「下载并更新」？更新时程序会短暂退出并自动重启。"
        )
        return native.util.ask_yes_no(message, title="xxl-whisper 更新")

    def _install(self, release: ReleaseInfo) -> None:
        """Download, verify, and hand off to the self-update helper."""
        assets = find_update_assets(release)
        if assets is None:  # the caller already confirmed applicability
            return
        try:
            checksum = fetch_checksum(assets.checksum.url)
            dest = updates_dir() / download_filename(release.tag)
            download_file(
                build_update_spec(assets.exe, checksum, dest),
                self._on_progress,
                proxy=self._deps.get_proxy(),
            )
        except (DownloadError, UpdateCheckError) as exc:
            self._report_failure(f"下载更新失败：{exc}")
            return
        self._deps.indicator.show("正在安装更新，程序即将重启…")
        if not launch_helper(dest, Path(sys.executable)):
            self._report_failure("无法启动更新程序，请手动下载新版本。")
            return
        log.info("update: helper launched, quitting for %s", release.tag)
        self._deps.on_quit()

    def _on_progress(self, filename: str, downloaded: int, total: int) -> None:
        pct = downloaded / total if total else 0.0
        self._deps.indicator.progress(pct, f"下载更新 {filename}")

    def _report_failure(self, reason: str) -> None:
        log.warning("update: %s", reason)
        self._deps.indicator.hide()
        native.util.show_info(reason, title="xxl-whisper 更新失败")

    def _offer(self, release: ReleaseInfo) -> bool:
        notes = f"\n\n{release.notes}" if release.notes else ""
        current_text = ".".join(map(str, self._current))
        message = (
            f"发现新版本 {release.tag}（当前 v{current_text}）。{notes}\n\n现在打开下载页？"
        )
        return native.util.ask_yes_no(message, title="xxl-whisper 更新")
