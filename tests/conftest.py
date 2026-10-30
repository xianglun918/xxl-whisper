"""Shared test setup: make the suite runnable without a desktop session.

Two platform layers used to make ``import app.*`` fail off Windows/macOS, which
turned CI on Linux into "collection error" instead of a test run:

* ``app.tray`` imports :mod:`pystray`, whose Xorg backend connects to the X
  server *at import time*. On a headless runner that raises
  ``Xlib.error.DisplayNameError``.
* pystray ships a ``dummy`` backend that still provides the real
  ``Menu``/``MenuItem`` semantics the tray-adapter tests pin.

So on a headless Linux box we select the dummy backend before anything imports
``app.tray``. Windows and macOS (and any machine with a display) are untouched.
"""

import os
import sys

if sys.platform not in {"win32", "darwin"} and not os.environ.get("DISPLAY"):
    os.environ.setdefault("PYSTRAY_BACKEND", "dummy")
