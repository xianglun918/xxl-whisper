"""Update flow: silent auto-check watcher plus a user-triggered dialog path.

Auto path never blocks or steals focus: it posts one tray notification when a
newer GitHub release appears. Manual path (tray menu) shows a version dialog
and opens the download page on confirmation. Network failures degrade to a
log line (auto) or a small info box (manual).
"""

import logging
import threading
import webbrowser

from app import winutil
from app.tray import Tray
from app.updater import ReleaseInfo, UpdateCheckError, fetch_latest_release, is_newer, parse_version

log = logging.getLogger(__name__)

_INITIAL_DELAY_S: float = 15.0
_RECHECK_INTERVAL_S: float = 24 * 3600.0


class UpdateFlow:
    """Owns the update-watcher thread and the manual check dialog."""

    def __init__(self, tray: Tray, current_version: str) -> None:
        self._tray = tray
        self._current = parse_version(current_version)
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
            winutil.show_info(f"已是最新版本 {current_text}。")
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
                self._tray.notify(
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

    def _offer(self, release: ReleaseInfo) -> bool:
        notes = f"\n\n{release.notes}" if release.notes else ""
        current_text = ".".join(map(str, self._current))
        message = (
            f"发现新版本 {release.tag}（当前 v{current_text}）。{notes}\n\n现在打开下载页？"
        )
        return winutil.ask_yes_no(message, title="xxl-whisper 更新")
