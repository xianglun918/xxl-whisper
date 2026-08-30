"""Entry point: logging, single instance, DPI awareness, config, app loop."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app import winutil
from app.app import DictationApp
from app.config import ConfigError, config_dir, config_path, load_config


def entry() -> None:
    """Process entry: never returns when the app runs to completion."""
    config_dir().mkdir(parents=True, exist_ok=True)
    _setup_logging()
    winutil.set_dpi_awareness()
    if not winutil.acquire_single_instance():
        winutil.show_error("xxl-whisper 已在运行（请查看托盘图标）。")
        sys.exit(1)
    try:
        config = load_config(config_path())
        DictationApp(config).run()
    except ConfigError as exc:
        logging.getLogger(__name__).exception("config error")
        winutil.show_error(f"配置文件错误：\n{exc}")
        sys.exit(1)
    except Exception:
        logging.getLogger(__name__).exception("fatal error")
        raise


def _setup_logging() -> None:
    log_dir = Path(config_dir()) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / "app.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(name)s %(message)s",
        handlers=[handler],
    )
