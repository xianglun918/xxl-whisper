"""Entry point: logging, single instance, DPI awareness, config, app loop."""

import faulthandler
import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import TracebackType
from typing import Any, cast

from app import native
from app.app import DictationApp
from app.config import ConfigError, config_dir, config_path, load_config
from app.selfupdate import parse_update_args, run_helper


def entry() -> None:
    """Process entry: never returns when the app runs to completion."""
    config_dir().mkdir(parents=True, exist_ok=True)
    _setup_logging()
    _install_crash_handlers()
    # The self-update helper runs before any normal startup: no tray, no config
    # prompt, and crucially no single-instance acquisition (it must observe the
    # running app release that mutex, not compete with it).
    request = parse_update_args(sys.argv[1:])
    if request is not None:
        run_helper(request)
        return
    native.util.set_dpi_awareness()
    if not native.util.acquire_single_instance():
        native.util.show_error("xxl-whisper 已在运行（请查看托盘图标）。")
        sys.exit(1)
    try:
        config = load_config(config_path())
    except ConfigError as exc:
        logging.getLogger(__name__).exception("config error")
        native.util.show_error(f"配置文件错误：\n{exc}")
        sys.exit(1)
    try:
        DictationApp(config).run()
    except Exception as exc:
        # A windowed build has no console: re-raising only ever produced a
        # silent vanishing tray icon. Record it, then say so on screen.
        logging.getLogger(__name__).exception("fatal error")
        _show_fatal(exc)
        raise


def _setup_logging() -> None:
    log_dir = Path(config_dir()) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / "app.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(name)s %(message)s",
        handlers=[handler],
    )


class _CrashFile:
    """Holds the crash log handle: faulthandler writes to it, so it must live on."""

    handle: object | None = None


def _install_crash_handlers() -> None:
    """Log every uncaught exception, thread death and fatal signal.

    Three silent failure modes used to leave nothing behind: an exception on the
    main thread, an exception in one of the daemon threads (decode, VAD, emit,
    hooks), and a hard crash in native code. All three now land in
    ``logs/app.log`` / ``logs/crash.log``, which is the only diagnostic a user of
    a windowed app can hand over.
    """
    log = logging.getLogger(__name__)

    def _hook(
        exc_type: type[BaseException],
        exc: BaseException,
        tb: TracebackType | None,
    ) -> None:
        log.critical("uncaught exception", exc_info=(exc_type, exc, tb))

    def _thread_hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is SystemExit:
            return
        log.critical(
            "uncaught exception in thread %r",
            args.thread.name if args.thread is not None else "?",
            exc_info=cast("Any", (args.exc_type, args.exc_value, args.exc_traceback)),
        )

    sys.excepthook = _hook
    threading.excepthook = _thread_hook
    try:
        _CrashFile.handle = (Path(config_dir()) / "logs" / "crash.log").open(
            "a", encoding="utf-8"
        )
        faulthandler.enable(file=_CrashFile.handle)
    except OSError as exc:
        log.warning("crash log unavailable: %s", exc)


def _show_fatal(exc: BaseException) -> None:
    """Best-effort dialog after a fatal error; never raises."""
    try:
        native.util.show_error(
            f"xxl-whisper 遇到错误已退出：\n{exc}\n\n日志：{config_dir() / 'logs' / 'app.log'}"
        )
    except Exception:
        logging.getLogger(__name__).exception("could not show the fatal-error dialog")
