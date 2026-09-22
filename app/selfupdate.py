"""Windows one-click self-update: helper process + atomic exe replacement.

Upgrading a portable onefile build means replacing the running ``.exe``, which
Windows locks while it runs. The strategy is: download the new build, copy the
*current* exe to a temp dir, launch that copy in ``--apply-update`` helper mode,
then quit. The helper waits for this process to release the single-instance
mutex, atomically swaps the new exe into place, relaunches the app, and cleans
itself up. The pure decision helpers below are platform-neutral and unit-tested;
only :func:`run_helper` touches the platform facade, and it is Windows-only —
macOS keeps the browser-based flow.
"""

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.download_io import FileSpec
from app.updater import AssetInfo

log = logging.getLogger(__name__)

#: Helper-mode flag: ``xxl-whisper.exe --apply-update <target> <new-exe>``.
APPLY_FLAG: str = "--apply-update"
#: The flag plus its two path arguments.
_APPLY_ARG_COUNT: int = 3

#: How long the helper waits for the app to release the single-instance mutex.
_EXIT_TIMEOUT_S: float = 60.0
_EXIT_POLL_S: float = 0.2
#: Replacement retries: a transient AV/indexer lock clears within a few seconds.
_REPLACE_ATTEMPTS: int = 20
_REPLACE_INTERVAL_S: float = 0.5

_CREATE_NO_WINDOW: int = 0x08000000
_DETACHED_PROCESS: int = 0x00000008


@dataclass(frozen=True, slots=True)
class ApplyRequest:
    """Parsed ``--apply-update`` invocation: replace *target* with *source*."""

    target: Path
    source: Path


@dataclass(frozen=True, slots=True)
class ReplaceOptions:
    """Retry policy plus injectable filesystem seams for :func:`install_new_exe`.

    The callables default to the real ``shutil``/``os``/``time`` functions and
    exist so the retry loop can be exercised in tests without any I/O.
    """

    attempts: int = _REPLACE_ATTEMPTS
    interval_s: float = _REPLACE_INTERVAL_S
    copy: Callable[[Path, Path], object] = shutil.copyfile
    replace: Callable[[Path, Path], None] = os.replace
    sleep: Callable[[float], None] = time.sleep


_DEFAULT_OPTIONS = ReplaceOptions()


def parse_update_args(argv: Sequence[str]) -> ApplyRequest | None:
    """Parse the helper flag; ``None`` when this is a normal app launch.

    Only the exact ``--apply-update <target> <new-exe>`` shape is a helper run;
    any other argument vector (including extra args) is treated as a normal
    launch, so a stray argument can never trigger an exe swap.
    """
    if len(argv) != _APPLY_ARG_COUNT or argv[0] != APPLY_FLAG:
        return None
    return ApplyRequest(target=Path(argv[1]), source=Path(argv[2]))


def download_filename(tag: str) -> str:
    """Local file name for a release's exe (``v0.6.2`` -> ``xxl-whisper-0.6.2.exe``)."""
    return f"xxl-whisper-{tag.strip().lstrip('vV')}.exe"


def build_update_spec(asset: AssetInfo, checksum: str, dest: Path) -> FileSpec:
    """Build a download_io spec that verifies the new exe against *checksum*."""
    return FileSpec(
        url=asset.url,
        dest=dest,
        expected_size=asset.size,
        expected_sha256=frozenset({checksum}),
    )


def supports_self_update() -> bool:
    """Whether this build can replace itself (frozen Windows onefile only)."""
    return sys.platform == "win32" and bool(getattr(sys, "frozen", False))


def should_retry_replace(attempt: int, attempts: int, exc: OSError) -> bool:
    """Whether a locked target is worth another replace attempt.

    A ``PermissionError`` (the exe still mapped by antivirus or a lingering
    handle) is transient; any other ``OSError`` (missing source, cross-volume
    move) is not, so the caller fails fast instead of spinning.
    """
    return attempt < attempts and isinstance(exc, PermissionError)


def wait_until_released(
    probe: Callable[[], bool],
    *,
    timeout_s: float,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Poll *probe* until it reports the instance gone; ``False`` on timeout."""
    deadline = time.monotonic() + timeout_s
    while probe():
        if time.monotonic() >= deadline:
            return False
        sleep(_EXIT_POLL_S)
    return True


def install_new_exe(
    source: Path,
    target: Path,
    *,
    options: ReplaceOptions = _DEFAULT_OPTIONS,
) -> None:
    """Atomically swap *source* over *target*, retrying while the target is locked.

    The new exe is staged beside the target (same volume, so ``os.replace`` is a
    real atomic rename and a download on another volume cannot fail the swap),
    then renamed over the target. Any failure removes the staging file, so the
    user is never left with a half-written exe.
    """
    staging = target.with_name(f".{target.name}.new-{os.getpid()}")
    try:
        options.copy(source, staging)
        _replace_with_retry(staging, target, options=options)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise


def _replace_with_retry(staging: Path, target: Path, *, options: ReplaceOptions) -> None:
    for attempt in range(1, options.attempts + 1):
        try:
            options.replace(staging, target)
        except OSError as exc:
            if not should_retry_replace(attempt, options.attempts, exc):
                raise
            log.warning(
                "update: target locked (attempt %d/%d): %s", attempt, options.attempts, exc
            )
            options.sleep(options.interval_s)
        else:
            return


def launch_helper(new_exe: Path, target: Path) -> bool:
    """Copy this running exe to a temp dir and launch it in helper mode."""
    if not supports_self_update():
        return False
    try:
        helper = _stage_helper_copy()
        subprocess.Popen(  # noqa: S603 — our own frozen exe, no shell
            [str(helper), APPLY_FLAG, str(target), str(new_exe)],
            close_fds=True,
            creationflags=_detached_flags(),
        )
    except OSError:
        log.exception("update: failed to launch the update helper")
        return False
    return True


def _stage_helper_copy() -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="xxl-whisper-update-"))
    helper = temp_dir / Path(sys.executable).name
    shutil.copy2(sys.executable, helper)
    return helper


def _detached_flags() -> int:
    """Fully detach a launched exe from this process (no shared console)."""
    return _DETACHED_PROCESS if sys.platform == "win32" else 0


def run_helper(request: ApplyRequest) -> None:
    """Helper-process entry: wait, swap, relaunch, clean up."""
    log.info("update helper: target=%s source=%s", request.target, request.source)
    if sys.platform != "win32":
        log.warning("update helper is Windows-only; ignoring")
        return
    from app import native  # noqa: PLC0415 — platform facade loaded only in helper mode

    if not wait_until_released(native.util.instance_running, timeout_s=_EXIT_TIMEOUT_S):
        log.error("update helper: the running instance did not exit in time")
        native.util.show_error("更新失败：等待旧版本退出超时。请重启程序后重试。")
        return
    try:
        install_new_exe(request.source, request.target)
    except OSError as exc:
        log.exception("update helper: could not replace the program file")
        native.util.show_error(f"更新失败：无法替换程序文件，已保留原版本。\n{exc}")
        _relaunch(request.target)  # keep the user's app running on the old build
        _cleanup_self(request.target)
        return
    if not _relaunch(request.target):
        native.util.show_error("更新已完成，但自动重启失败，请手动启动 xxl-whisper。")
    _cleanup_self(request.target)


def _relaunch(target: Path) -> bool:
    try:
        subprocess.Popen(  # noqa: S603 — our own frozen exe, no shell
            [str(target)], close_fds=True, creationflags=_detached_flags()
        )
    except OSError:
        log.exception("update helper: relaunch failed")
        return False
    return True


def _cleanup_self(target: Path) -> None:
    """Remove the temp helper copy; a running exe needs a deferred delete."""
    self_path = Path(sys.executable)
    if self_path == target:
        return
    try:
        self_path.unlink()
    except OSError:
        pass
    else:
        return
    _schedule_delete(self_path)


def _schedule_delete(path: Path) -> None:
    """Delete a still-locked file via a transient windowless shell."""
    comspec = os.environ.get("COMSPEC") or "C:\\Windows\\System32\\cmd.exe"
    command = f'ping -n 3 127.0.0.1 >nul & del /f /q "{path}" & rmdir /q "{path.parent}"'
    try:
        subprocess.Popen(  # noqa: S603 — fixed COMSPEC, no shell
            [comspec, "/c", command], close_fds=True, creationflags=_CREATE_NO_WINDOW
        )
    except OSError:
        log.warning("update helper: could not schedule removal of %s", path)
