# xxl-whisper

> 🌐 **中文** · [English](README.en.md)

Windows 托盘常驻的离线语音听写工具：**按住 CapsLock 说话，松开后中文直接上屏到光标处**。

> 📖 **[使用与分发说明](docs/使用与分发说明.md)** — 给使用者的安装/常见问题一页纸，给维护者的打包/发版流程。

## 系统架构

[![xxl-whisper 架构图](docs/architecture.visual-check.1440x900.light.png)](https://xianglun918.github.io/xxl-whisper/architecture.html)

**主链路**：HotkeyHook（CapsLock 拦截）→ HoldClickDetector（按住/单击判定）→ Recorder（16 kHz 门控录音）→ Recognizer（SenseVoice 离线识别）→ Emit（三通道上屏：注入 Ctrl+V → WM_PASTE → 剪贴板提示）→ 目标应用光标处。

👆 点击图片打开**交互式架构图**（明暗主题、缩放、聚焦、导出；JSON 源在 `docs/architecture.json`）。

- 纯本地识别（SenseVoice-Small，CPU 实时 25 倍速），无网可用，不依赖任何云服务和网盘
- 中英混说（"开个 PR"、"看下 README"）
- 单击 CapsLock 仍是大小写切换（原功能不破坏）
- 托盘菜单：暂停热键 / 选麦克风 / **换热键（CapsLock、F2-F8、Scroll Lock、鼠标侧键 X1/X2、自定义任意键）** / 换模型（SenseVoice-Small / FunASR-Nano） / 开机自启 / 检查更新 / 退出
- 版本升级：启动与每 24 小时静默检查 GitHub Release，新版托盘提示，一键直达下载页（`check_updates` 可关）
- 上屏通道自动降级：键盘注入 → WM_PASTE → UIA → 剪贴板提示
- 语义顺滑：托盘一键开关，去「嗯/呃/啊」语气词、重复、口误（Fun-ASR-Nano，LLM 解码器原生能力；切到该模型时默认打开）

## 首次使用（最终用户）

1. 拿到 `xxl-whisper.exe`，双击运行（首次会自动从 hf-mirror 下载约 230MB 模型，有进度提示）
2. 托盘出现图标即就绪。按住 CapsLock 说话 → 松开 → 文字出现在当前输入焦点
3. 右键托盘图标可暂停、换麦克风、开机自启

配置文件：`%LOCALAPPDATA%\xxl-whisper\config.toml`（热键/阈值/线程数等，改完重启生效）
日志：`%LOCALAPPDATA%\xxl-whisper\logs\app.log`

## 开发

```bash
uv sync                     # 建虚拟环境装依赖
uv run pytest tests -q      # 单测 + 集成测试（需已下载模型）
uv run python run.py        # 源码运行
build.bat                   # PyInstaller 打包 -> dist\xxl-whisper.exe
```

## 架构

详见上方交互式架构图（[线上版](https://xianglun918.github.io/xxl-whisper/architecture.html)，GitHub Pages 托管）。线程模型：主线程跑托盘；钩子线程泵
Win32 消息；PortAudio 回调收音；单一 ASR worker 消费消息队列（热键事件与控制命令共用
一个 tagged union 队列，无锁竞争）。

## 模型

- **SenseVoice-Small**（默认）：INT8 ONNX（239,233,841 字节）+ tokens.txt（315,894 字节）
- **Fun-ASR-Nano**（可选，推荐新模型，精度更高）：encoder_adaptor / embedding / llm
  三个 int8 ONNX（约 993MB）+ Qwen3-0.6B 分词器，中英混说与方言表现更优；
  因是 LLM 解码器，支持「语义顺滑」（去语气词/重复/口误），托盘可一键开关
- 来源：`hf-mirror.com`（主源，国内直连）→ GitHub release tar.bz2 兜底
- 存放：`%LOCALAPPDATA%\xxl-whisper\models\<模型名>\`
- 注意：modelscope 的 iic/SenseVoiceSmall-onnx 与 sherpa-onnx 轮子内置的 onnxruntime
  不兼容（ORT API 版本冲突），不要混用

### 代理配置（内网环境）

若你的网络需要代理才能访问外网，在配置文件中添加 `proxy` 项即可（重启生效）：

```toml
# %LOCALAPPDATA%\xxl-whisper\config.toml
proxy = "http://你的代理地址:端口"
```

也可通过环境变量 `HTTPS_PROXY` / `HTTP_PROXY` 配置；配置文件中的 `proxy` 优先级高于环境变量；留空则使用环境变量/系统代理。

### 手动下载模型（内网 / 代理受限时）

软件会自动下载模型；若网络受限导致下载失败，程序会弹窗提示。此时可手动访问以下
URL 下载文件，放入对应目录（**子目录需按路径自行创建**，下载后重启软件即可）：

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

## 已知边界

- 管理员窗口（提权的终端/软件）里上屏会被 UIPI 拦截 → 不支持
- **本机若有软件拦截键盘注入**（uTools/豆包/ArmouryCrate/G HUB 等带全局钩子的
  工具可能如此，症状：日志出现"本机拦截键盘注入"），自动回退为 WM_PASTE
  定点粘贴；若目标程序也不支持，文字保留在剪贴板并提示手动 Ctrl+V
- 杀软误报 Python 打包 exe：加白即可

## macOS 版

macOS 版与 Windows 版功能对等：菜单栏常驻的离线听写工具，**按住右 Command 说话，松开后文字直接上屏到光标处**。识别同样纯本地完成（SenseVoice-Small / Fun-ASR-Nano，sherpa-onnx）。仅支持 Apple 芯片（arm64），需 **macOS 15 或更高**。

### 安装

1. 从 GitHub Release 下载 `xxl-whisper-arm64.zip`，解压得到 `xxl-whisper.app`
2. 把 `.app` 放到任意固定目录（如「应用程序」）
3. **首次打开：右键点图标 → 选「打开」**（Control + 单击等同右键）。当前构建是 ad-hoc 签名、未公证，双击会被 Gatekeeper 拦下；右键打开一次后即可正常双击

首次运行会自动从 hf-mirror 下载约 230MB 模型（同 Windows 版）。需 macOS 15+ 的原因：sherpa-onnx 的预编译库在 macOS 14 及以下会直接加载失败（上游 issue [k2-fsa/sherpa-onnx#3840](https://github.com/k2-fsa/sherpa-onnx/issues/3840)）。

### 首次授权（三项，必须）

mac 与 Windows 最大的不同是需要三项系统授权，缺一不可。应用启动后会检测并弹窗引导，点「打开设置」会深链到对应面板：

| 权限 | 用途 | 系统设置位置 |
|---|---|---|
| 辅助功能 | 注入 Cmd+V 上屏 | 隐私与安全性 → 辅助功能 |
| 输入监控 | 监听热键 | 隐私与安全性 → 输入监控 |
| 麦克风 | 录音 | 首次录音时系统弹窗授权 |

**每次更新后，辅助功能与输入监控都要重新勾选一次。** ad-hoc 构建按二进制哈希识别身份，更新后旧授权失效；应用启动时会自动清理过期的授权记录（`tccutil reset`），因此系统设置里复选框显示为**未勾选**，每项**勾一次**即可，不用先删再添加。将来换成 Developer ID + 公证构建后这一步摩擦会消失（尚未实现）。

> 源码运行（非 .app）时，授权归属宿主终端（如 iTerm2）；用 `tccutil reset` 可清掉重测。

### 热键

默认热键是**右 Command**（mac 无 CapsLock 选项，也没有 Scroll Lock 与鼠标侧键）。菜单栏图标 → 热键可切换：右 Command / F2 / F4 / F6 / F8 / 自定义按键（点后按任意键，Esc 取消）。

> mac 的 F 键默认是媒体键（亮度、音量等）；要用 F2/F4/F6/F8 当热键，需按住 Fn，或在「系统设置 → 键盘」里开启「将 F1、F2 等键用作标准功能键」。

### 配置与日志

- 配置：`~/Library/Application Support/xxl-whisper/config.toml`（改完重启生效）
- 日志：`~/Library/Application Support/xxl-whisper/logs/app.log`
- 模型：`~/Library/Application Support/xxl-whisper/models/`

### 开发

```bash
uv sync                     # 建虚拟环境装依赖
uv run pytest tests -q      # 单测 + 集成测试（需已下载模型）
uv run python run.py        # 源码运行
bash build.sh               # PyInstaller 打包 -> dist/xxl-whisper.app + dist/xxl-whisper-arm64.zip
```

`build.sh` 需要仓库内的 `.venv`（先 `uv sync`）。默认 ad-hoc 签名；`CODESIGN_IDENTITY="Developer ID Application: ..." bash build.sh` 可切换 Developer ID 签名（供将来的公证使用）。

### macOS 已知边界

- 需 **macOS 15+（arm64）**：sherpa-onnx 预编译库在 14 及以下加载即崩（上游 #3840）
- **首次打开需右键 → 打开**：ad-hoc 签名、未公证，被 Gatekeeper 拦；与 Windows 版「杀软误报加白」同量级
- **每次更新后辅助功能 / 输入监控需重新勾选**：ad-hoc 构建按二进制哈希识别身份；应用启动会自动 `tccutil reset` 清掉过期记录，系统设置里显示未勾选，每项勾一次即可
- 默认热键右 Command，无 CapsLock 选项；F 键预设需按 Fn 或改系统「标准功能键」设置
- 上屏为 2 通道（Cmd+V 注入 → 剪贴板提示），无 Windows 版的 WM_PASTE / UIA 通道
- 「正在听…」条位于屏幕底部居中；当前台窗口铺满屏幕时自动隐藏（近似判断，最大化的普通窗口也会隐藏）
- 系统麦克风指示灯只在按住热键期间亮起（mac 上每次按住才开录音流，实测开流约 68ms）
- 「开机自启」在源码运行下不可用（需打包后的 .app）；菜单项显示未勾选并记一条警告日志
- 托盘的单选菜单显示为勾选（pystray-mac 不支持圆点样式，行为无差异）
- 上屏前会先把文字写入剪贴板（与 Windows 版相同）
