import pathlib


def edit(path, pairs):
    p = pathlib.Path(path)
    s = p.read_text()
    for old, new in pairs:
        if old not in s:
            raise SystemExit(f"MISS in {path}: {old[:90]!r}")
        s = s.replace(old, new, 1)
    p.write_text(s)
    print("ok", path)


# ── ① recorder: 开流幂等 + start/stop 原子化 + 超限即丢弃 ──────────────────
edit(
    "app/recorder.py",
    [
        (
            "        self._lock = threading.Lock()\n",
            "        self._lock = threading.Lock()\n"
            "        #: Serialises the open/close decisions against each other: the hook\n"
            "        #: thread and the worker can both call start()/stop() for one hold, and\n"
            "        #: \"check → open → mark\" is not atomic on its own.\n"
            "        self._lifecycle = threading.Lock()\n",
        ),
        (
            '''    def _open_stream(self) -> None:
        if self._source == "system":''',
            '''    def _open_stream(self) -> None:
        """Open the capture stream; a second call is a no-op.

        Idempotent on purpose: macOS opens a stream per hold, so a double open
        meant a leaked stream and a lit microphone indicator for every sentence.
        """
        if self._stream is not None:
            return
        if self._source == "system":''',
        ),
        (
            """        if self._capturing_hold():
            return
        if not native.RECORDER_KEEP_OPEN:
            self._open_stream()
        with self._lock:
            if self._recording:
                return
            self._buffer = []
            self._buffered_samples = 0
            self._recording = True""",
            """        with self._lifecycle:
            if self._capturing_hold():
                return
            if not native.RECORDER_KEEP_OPEN:
                self._open_stream()
            with self._lock:
                if self._recording:
                    return
                self._buffer = []
                self._buffered_samples = 0
                self._recording = True""",
        ),
        (
            """        with self._lock:
            self._recording = False
            chunks = self._buffer
            self._buffer = []
            self._buffered_samples = 0
        if not native.RECORDER_KEEP_OPEN:
            self._close_stream()""",
            """        with self._lifecycle:
            with self._lock:
                self._recording = False
                chunks = self._buffer
                self._buffer = []
                self._buffered_samples = 0
            if not native.RECORDER_KEEP_OPEN:
                self._close_stream()""",
        ),
        (
            """        audio = np.concatenate(chunks, axis=0).reshape(-1)
        if audio.shape[0] < SAMPLE_RATE * MIN_RECORD_MS // 1000:
            return None
        return audio""",
            """        audio = np.concatenate(chunks, axis=0).reshape(-1)
        if audio.shape[0] > _MAX_HOLD_SAMPLES:
            # Nobody holds a hotkey for two minutes: this capture was never
            # stopped (a skipped press, a lost key-up). Shipping it to the model
            # would paste minutes of stale audio, so drop it instead of "capping".
            log.warning(
                "recorder: dropped a runaway capture (%.0f s)", audio.shape[0] / SAMPLE_RATE
            )
            return None
        if audio.shape[0] < SAMPLE_RATE * MIN_RECORD_MS // 1000:
            return None
        return audio""",
        ),
    ],
)

# ── ③④⑤ emit: 兜底不还原、写成功后再释放、跨 burst 刷新快照 ────────────────
edit(
    "app/emit.py",
    [
        (
            """    original: object | None = None
    generation: int = 0
    timer: threading.Timer | None = None""",
            """    original: object | None = None
    #: What we last wrote to the clipboard. Used to tell "still our own text"
    #: (the snapshot is valid) from "the user copied something new" (it is not).
    last_staged: object | None = None
    generation: int = 0
    timer: threading.Timer | None = None""",
        ),
        (
            '''def _snapshot_original() -> None:
    """Remember the user's clipboard once, before this burst's first stage."""
    with _RESTORE_LOCK:
        if _STATE.original is not None:
            return
    original = native.io.snapshot_clipboard()
    with _RESTORE_LOCK:
        # Re-check: two emits can race here, and the first snapshot wins.
        if _STATE.original is None:
            _STATE.original = original''',
            '''def _snapshot_original() -> None:
    """Remember the user's clipboard, refreshing it when it changed under us.

    Within one burst the clipboard holds our own staged text, so the snapshot
    taken before the first sentence stays valid and must be kept — re-reading it
    would capture the previous sentence. Across bursts the user may have copied
    something new, and the stale snapshot would then overwrite it, so a changed
    clipboard replaces the snapshot instead of being ignored.
    """
    current = native.io.snapshot_clipboard()
    with _RESTORE_LOCK:
        if _STATE.original is not None and current == _STATE.last_staged:
            return  # still our own staged text: keep the burst's snapshot
        _STATE.original = current''',
        ),
        (
            """    with _RESTORE_LOCK:
        if _STATE.original is not None:
            return""",
            """    with _RESTORE_LOCK:
        if _STATE.original is not None:
            return""",
        ),
        (
            """    if not settings.restore_clipboard:
        return  # the user asked to keep the dictated text on the clipboard
    if delivered is Channel.CLIPBOARD and not clipboard_read:
        # Terminal fallback: the text has to stay there, otherwise the "请手动
        # 粘贴" hint would paste whatever the user had copied before.
        return
    _restore_clipboard_later(settings.paste_delay_ms if clipboard_read else 0)""",
            """    if not settings.restore_clipboard or delivered is Channel.CLIPBOARD:
        # Either the user asked to keep the dictated text, or nothing was
        # delivered at all: in both cases the text must stay on the clipboard,
        # otherwise the "请手动粘贴" hint would paste whatever was there before.
        # The snapshot is useless now (the original is already gone), and keeping
        # it would let it overwrite a *later* copy the user makes.
        _forget_original()
        return
    _restore_clipboard_later(settings.paste_delay_ms if clipboard_read else 0)


def _forget_original() -> None:
    \"\"\"Drop the snapshot after it can no longer be restored meaningfully.\"\"\"
    with _RESTORE_LOCK:
        _STATE.original = None
        _STATE.last_staged = None
        _STATE.generation += 1
        if _STATE.timer is not None:
            _STATE.timer.cancel()
            _STATE.timer = None""",
        ),
        (
            """def _do_restore(generation: int) -> None:
    \"\"\"Write the user's clipboard back unless a newer sentence took over.\"\"\"
    with _RESTORE_LOCK:
        if generation != _STATE.generation:
            return
        original = _STATE.original
        _STATE.original = None
        _STATE.timer = None
    if original is None:
        return
    try:
        native.io.restore_clipboard(original)
    except Exception:
        log.exception("emit: clipboard restore failed")""",
            """def _do_restore(generation: int) -> None:
    \"\"\"Write the user's clipboard back unless a newer sentence took over.

    The snapshot is released **after** the write, not before: releasing first left
    a window (Win32 ``OpenClipboard`` retries for up to ~1 s) in which the next
    sentence snapshotted our own text as "the user's clipboard" — the original was
    then lost for good.
    \"\"\"
    with _RESTORE_LOCK:
        if generation != _STATE.generation:
            return
        original = _STATE.original
    if original is None:
        with _RESTORE_LOCK:
            if generation == _STATE.generation:
                _STATE.timer = None
        return
    try:
        native.io.restore_clipboard(original)
    except Exception:
        log.exception("emit: clipboard restore failed")
        return
    with _RESTORE_LOCK:
        if generation == _STATE.generation:
            _STATE.original = None
            _STATE.last_staged = None
            _STATE.timer = None""",
        ),
        (
            """    # A restore queued by the previous sentence is stale now: this call is about
    # to overwrite the clipboard again. The *original* snapshot stays.
    _bump_generation()
    try:""",
            """    # A restore queued by the previous sentence is stale now: this call is about
    # to overwrite the clipboard again. The *original* snapshot stays.
    _bump_generation()
    with _RESTORE_LOCK:
        _STATE.last_staged = text
    try:""",
        ),
    ],
)
print("done")
