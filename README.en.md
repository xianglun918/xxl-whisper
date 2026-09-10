# xxl-whisper

> 🌐 [中文](README.md) · **English**

An offline voice dictation tool that lives in the Windows system tray: **hold CapsLock to talk, release, and your Chinese is inserted directly at the cursor**.

> 📖 **[Usage and Distribution Guide](docs/使用与分发说明.en.md)** — a one-pager on installation and common questions for users, and the build/release process for maintainers.

## System Architecture

[![xxl-whisper architecture diagram](docs/architecture.visual-check.1440x900.light.png)](https://xianglun918.github.io/xxl-whisper/architecture.html)

**Main pipeline**: HotkeyHook (CapsLock interception) → HoldClickDetector (hold/click detection) → Recorder (16 kHz gated recording) → Recognizer (SenseVoice offline recognition) → Emit (three-channel insertion: Ctrl+V injection → WM_PASTE → clipboard prompt) → the cursor in the target app.

👆 Click the image to open the **interactive architecture diagram** (light/dark themes, zoom, focus, export; JSON source in `docs/architecture.json`).

- Fully local recognition (SenseVoice-Small, 25x realtime on CPU), works with no network, no cloud services or cloud storage required
- Mixed Chinese–English speech ("open a PR", "check the README")
- A single CapsLock tap still toggles case (original behavior preserved)
- System tray menu: pause hotkey / pick microphone / **change hotkey (CapsLock, F2-F8, Scroll Lock, mouse side buttons X1/X2, any custom key)** / switch model (SenseVoice-Small / FunASR-Nano) / launch at login / check for updates / quit
- Version updates: silently checks GitHub Releases at startup and every 24 hours; the tray notifies you of a new version with one-click access to the download page (`check_updates` can turn it off)
- Automatic insertion-channel fallback: keyboard injection → WM_PASTE → UIA → clipboard prompt
- Semantic smoothing: one-click toggle in the tray; strips "um/uh/ah" filler words, repetitions, and slips of the tongue (Fun-ASR-Nano, a native capability of its LLM decoder; enabled by default when you switch to that model)

## First Use (end users)

1. Get `xxl-whisper.exe` and double-click it (on first run it automatically downloads a ~230MB model from hf-mirror, with a progress indicator)
2. Once the tray icon appears, it's ready. Hold CapsLock and talk → release → the text appears at the current input focus
3. Right-click the tray icon to pause, change microphone, or launch at login

Config file: `%LOCALAPPDATA%\xxl-whisper\config.toml` (hotkey/thresholds/thread count, etc.; changes take effect after restart)
Log: `%LOCALAPPDATA%\xxl-whisper\logs\app.log`

## Development

```bash
uv sync                     # create the virtual environment and install dependencies
uv run pytest tests -q      # unit + integration tests (requires the model to be downloaded)
uv run python run.py        # run from source
build.bat                   # PyInstaller build -> dist\xxl-whisper.exe
```

## Architecture

See the interactive architecture diagram above ([online version](https://xianglun918.github.io/xxl-whisper/architecture.html), hosted on GitHub Pages). Threading model: the main thread runs the tray; a hook thread pumps
Win32 messages; the PortAudio callback captures audio; a single ASR worker consumes the message queue (hotkey events and control commands share
one tagged-union queue, with no lock contention).

## Models

- **SenseVoice-Small** (default): INT8 ONNX (239,233,841 bytes) + tokens.txt (315,894 bytes)
- **Fun-ASR-Nano** (optional, the recommended newer model, higher accuracy): encoder_adaptor / embedding / llm
  three int8 ONNX files (about 993MB) + the Qwen3-0.6B tokenizer; better at mixed Chinese–English speech and dialects;
  because it's an LLM decoder, it supports "semantic smoothing" (removing filler words/repetitions/slips of the tongue), toggleable with one click in the tray
- Source: `hf-mirror.com` (primary source, direct connection in mainland China) → GitHub release tar.bz2 as a fallback
- Location: `%LOCALAPPDATA%\xxl-whisper\models\<model name>\`
- Note: modelscope's iic/SenseVoiceSmall-onnx is incompatible with the onnxruntime
  bundled in the sherpa-onnx wheel (ORT API version conflict), so don't mix them

### Proxy Configuration (intranet environments)

If your network needs a proxy to reach the internet, just add a `proxy` entry to the config file (takes effect after restart):

```toml
# %LOCALAPPDATA%\xxl-whisper\config.toml
proxy = "http://your-proxy-address:port"
```

You can also configure it via the `HTTPS_PROXY` / `HTTP_PROXY` environment variables; the `proxy` in the config file takes precedence over the environment variables; if left empty, the environment variables/system proxy are used.

### Manual Model Download (intranet / restricted proxy)

The app downloads models automatically; if a restricted network makes the download fail, it shows a popup. You can then manually visit the
URLs below to download the files and place them in the matching directories (**create the subdirectories yourself to match the paths**; restart the app after downloading):

**SenseVoice-Small** → `%LOCALAPPDATA%\xxl-whisper\models\sensevoice\`

- `model.onnx` ← `https://hf-mirror.com/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/model.int8.onnx`
- `tokens.txt` ← `https://hf-mirror.com/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/tokens.txt`

**Fun-ASR-Nano** → `%LOCALAPPDATA%\xxl-whisper\models\funasr_nano\`

- `encoder_adaptor.int8.onnx` ← `https://hf-mirror.com/csukuangfj/sherpa-onnx-funasr-nano-int8-2025-12-30/resolve/main/encoder_adaptor.int8.onnx`
- `embedding.int8.onnx` ← `https://hf-mirror.com/csukuangfj/sherpa-onnx-funasr-nano-int8-2025-12-30/resolve/main/embedding.int8.onnx`
- `llm.int8.onnx` ← `https://hf-mirror.com/csukuangfj/sherpa-onnx-funasr-nano-int8-2025-12-30/resolve/main/llm.int8.onnx`
- `Qwen3-0.6B/merges.txt` ← `https://hf-mirror.com/csukuangfj/sherpa-onnx-funasr-nano-int8-2025-12-30/resolve/main/Qwen3-0.6B/merges.txt`
- `Qwen3-0.6B/tokenizer.json` ← `https://hf-mirror.com/csukuangfj/sherpa-onnx-funasr-nano-int8-2025-12-30/resolve/main/Qwen3-0.6B/tokenizer.json`
- `Qwen3-0.6B/vocab.json` ← `https://hf-mirror.com/csukuangfj/sherpa-onnx-funasr-nano-int8-2025-12-30/resolve/main/Qwen3-0.6B/vocab.json`

## Known Limitations

- Insertion in administrator windows (elevated terminals/apps) is blocked by UIPI → not supported
- **If software on your machine blocks keyboard injection** (tools with global hooks such as
  uTools/豆包/ArmouryCrate/G HUB may do this; symptom: "local keyboard injection blocked" in the log), it automatically falls back to WM_PASTE
  at the cursor; if the target app doesn't support that either, the text stays on the clipboard with a prompt to press Ctrl+V manually
- Antivirus false positives on the Python-packaged exe: just whitelist it

## macOS Version

The macOS version matches the Windows version feature-for-feature: an offline dictation tool that lives in the menu bar, **hold the right Command key to talk, release, and the text is inserted directly at the cursor**. Recognition is likewise done fully locally (SenseVoice-Small / Fun-ASR-Nano, sherpa-onnx). Apple silicon only (arm64), and it requires **macOS 15 or later**.

### Installation

1. Download `xxl-whisper-arm64.zip` from the GitHub Release and unzip it to get `xxl-whisper.app`
2. Put the `.app` in any permanent directory (such as Applications)
3. **First launch: right-click the icon → choose "Open"** (Control + click is the same as right-click). The current build is ad-hoc signed and not notarized, so double-clicking is blocked by Gatekeeper; after opening it once via right-click, you can double-click normally

The first run automatically downloads a ~230MB model from hf-mirror (same as the Windows version). Why macOS 15+ is required: sherpa-onnx's prebuilt library fails to load on macOS 14 and below (upstream issue [k2-fsa/sherpa-onnx#3840](https://github.com/k2-fsa/sherpa-onnx/issues/3840)).

### First-Time Permissions (three, mandatory)

The biggest difference from Windows is that macOS requires three system permissions, all of which are mandatory. After launch the app detects them and guides you with a popup; clicking "Open Settings" deep-links to the corresponding pane:

| Permission | Purpose | System Settings location |
|---|---|---|
| Accessibility | Inject Cmd+V to insert text | Privacy & Security → Accessibility |
| Input Monitoring | Listen for the hotkey | Privacy & Security → Input Monitoring |
| Microphone | Recording | System prompts for permission on first recording |

**After every update, Accessibility and Input Monitoring must be re-checked once.** An ad-hoc build is identified by its binary hash, so old permissions become invalid after an update; at launch the app automatically clears stale permission records (`tccutil reset`), so the checkboxes in System Settings appear **unchecked** and you just need to **check each one once**, with no need to remove and re-add first. This friction will disappear once we move to a Developer ID + notarized build (not implemented yet).

> When running from source (not the .app), permissions belong to the host terminal (such as iTerm2); `tccutil reset` can clear them so you can test again.

### Hotkey

The default hotkey is **right Command** (macOS has no CapsLock option, nor Scroll Lock or mouse side buttons). Menu bar icon → Hotkey lets you switch: right Command / F2 / F4 / F6 / F8 / custom key (click, then press any key; Esc cancels).

> On macOS the F keys are media keys by default (brightness, volume, etc.); to use F2/F4/F6/F8 as hotkeys, hold Fn, or enable "Use F1, F2, etc. keys as standard function keys" in System Settings → Keyboard.

### Config and Logs

- Config: `~/Library/Application Support/xxl-whisper/config.toml` (takes effect after restart)
- Log: `~/Library/Application Support/xxl-whisper/logs/app.log`
- Models: `~/Library/Application Support/xxl-whisper/models/`

### Development

```bash
uv sync                     # create the virtual environment and install dependencies
uv run pytest tests -q      # unit + integration tests (requires the model to be downloaded)
uv run python run.py        # run from source
bash build.sh               # PyInstaller build -> dist/xxl-whisper.app + dist/xxl-whisper-arm64.zip
```

`build.sh` needs the repo's `.venv` (run `uv sync` first). Ad-hoc signing by default; `CODESIGN_IDENTITY="Developer ID Application: ..." bash build.sh` switches to Developer ID signing (for future notarization).

### macOS Known Limitations

- Requires **macOS 15+ (arm64)**: sherpa-onnx's prebuilt library crashes on load on 14 and below (upstream #3840)
- **First launch requires right-click → Open**: ad-hoc signed, not notarized, blocked by Gatekeeper; same magnitude as the Windows version's "antivirus false positive, just whitelist"
- **After every update, Accessibility / Input Monitoring must be re-checked**: an ad-hoc build is identified by binary hash; at launch the app automatically runs `tccutil reset` to clear stale records, System Settings shows them unchecked, and you just check each one once
- Default hotkey is right Command, with no CapsLock option; the F-key presets require holding Fn or changing the system "standard function keys" setting
- Insertion uses 2 channels (Cmd+V injection → clipboard prompt), with no WM_PASTE / UIA channels like the Windows version
- The "Listening…" bar sits centered at the bottom of the screen; it auto-hides when the foreground window fills the screen (an approximation, so a maximized normal window also hides it)
- The system microphone indicator lights up only while the hotkey is held (on macOS the recording stream opens on each hold, measured at about 68ms to open)
- "Launch at login" is unavailable when running from source (it needs the packaged .app); the menu item shows unchecked and logs a warning
- The tray's radio menu items render as checkmarks (pystray-mac doesn't support the dot style; behavior is identical)
- Text is written to the clipboard before insertion (same as the Windows version)
