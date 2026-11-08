# xxl-whisper

> 🌐 [中文](README.md) · **English**

[![Website](https://img.shields.io/badge/Website-xianglun918.github.io-0a72ef?style=for-the-badge)](https://xianglun918.github.io/xxl-whisper/)
[![Latest release](https://img.shields.io/github/v/release/xianglun918/xxl-whisper?style=for-the-badge&label=Download&color=171717)](https://github.com/xianglun918/xxl-whisper/releases/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-4d4d4d?style=for-the-badge)](https://github.com/xianglun918/xxl-whisper/blob/main/LICENSE)

A dual-platform (Windows/macOS) offline voice dictation tool: hold the hotkey (CapsLock on Windows / Right Command on macOS), speak, release and the text is inserted at the cursor. It runs fully local with no network, and handles mixed Chinese–English speech.

[![xxl-whisper architecture](docs/architecture.en.png)](https://xianglun918.github.io/xxl-whisper/architecture.en.html)

## Quick start

1. **Download** — [Windows `xxl-whisper.exe`](https://github.com/xianglun918/xxl-whisper/releases/latest) or [macOS `xxl-whisper-arm64.zip`](https://github.com/xianglun918/xxl-whisper/releases/latest) (macOS: Apple silicon + 15 or later; first launch is right-click → Open)
2. **Run** — the first launch auto-downloads the speech model (~230MB, with a progress indicator)
3. **Use** — hold the hotkey, speak, release; a single CapsLock tap still toggles case

## Features

- Fully local, offline recognition (SenseVoice-Small / Fun-ASR-Nano) — works with no network
- Mixed Chinese–English speech ("open a PR", "check the README")
- Automatic insertion fallback for every kind of input field (4 levels on Windows / 2 on macOS)
- Hotkeys your way: CapsLock / F keys / mouse side buttons / custom
- Semantic smoothing: one click removes "um/uh/ah" fillers, repetitions and slips (Fun-ASR-Nano)
- Continuous dictation: always-on and key-free — a VAD splits sentences and inserts each one as you speak, ignoring silence and noise
- The indicator never steals focus and auto-hides in exclusive fullscreen; silent update checks on launch

## Platform support

| Capability | Windows (primary) | macOS (personal use) |
|---|---|---|
| Hold-hotkey dictation | ✅ | ✅ |
| Status indicator | ✅ Tk pill | ✅ native AppKit status bar |
| Subtitle overlay | ✅ Tk window | ✅ native AppKit window (no Tk) |
| System-audio input | ✅ WASAPI loopback, no driver | ⚠️ requires the [BlackHole](https://existential.audio/blackhole/) virtual device |
| Offline translation | ✅ | ✅ |
| Continuous dictation | ✅ | ✅ |
| File transcription | ✅ needs `ffmpeg` | ✅ needs `ffmpeg` |

**Dependency boundary**: Tk (`tkinter`) is Windows-only (status pill + subtitle overlay). macOS uses AppKit throughout (`NSStatusItem` status bar + native subtitle window) and **does not depend on Tk**; the macOS bundle ships no Tk, the Windows bundle ships it.

**Known limitations**

- Transcribing system/meeting audio on macOS: there is no loopback API — install [BlackHole](https://existential.audio/blackhole/) and select it as the input device.
- File transcription: depends on `ffmpeg` (a feature-time, not required, dependency); it is detected and prompted for only when that feature is used.
- macOS: Apple silicon (arm64) and macOS 15+ only.

## Documentation

| | |
|---|---|
| 🌐 Website | https://xianglun918.github.io/xxl-whisper/ |
| 📖 Usage & Distribution Guide | [docs/使用与分发说明.en.md](docs/使用与分发说明.en.md) — install, FAQ, config, models / proxy / manual download, known limits, macOS details, maintainer release |
| 🗺️ Interactive architecture | [online](https://xianglun918.github.io/xxl-whisper/architecture.en.html) · [JSON source](docs/architecture.en.json) |
| 🎬 Continuous dictation (visual) | [docs/continuous.en.html](docs/continuous.en.html) — animated walkthrough: how VAD gates speech and why silence/noise never produces text |

## Development

```bash
uv sync                  # create the virtualenv and install dependencies
uv run pytest tests -q   # unit + integration tests (model must be downloaded)
uv run python run.py     # run from source
build.bat                # Windows build -> dist\xxl-whisper.exe
bash build.sh            # macOS build -> dist/xxl-whisper-arm64.zip
```

## License

MIT
