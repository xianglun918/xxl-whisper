"""pystray macOS menu update must run on the main thread (P0 crash fix)."""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only pystray patch")


def test_pystray_menu_update_is_patched_to_main_thread() -> None:
    """Importing app.tray must replace pystray's off-main-thread setMenu_."""

    import app.tray  # noqa: PLC0415, F401 - import side effect applies the patch
    import pystray._darwin as darwin  # noqa: PLC0415 - imported after the skip guard

    patched = darwin.Icon._update_menu
    assert patched.__module__ == "app.tray", (
        "pystray._darwin.Icon._update_menu was not patched by app.tray"
    )
