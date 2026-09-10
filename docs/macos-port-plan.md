# xxl-whisper macOS 支持方案（决策记录与施工图）

> 状态：**决策已定稿，施工未开始**（2026-09-10 grilling 会话产出）。
> 分支：`feat/macos-support`。本文件是该分支的权威依据与施工图，后续会话按此作业；
> 若实现期发现与事实冲突，先回本文件修订决策记录，再动代码。

## 0. 背景与目标

xxl-whisper 是 Windows 托盘常驻的离线语音听写工具（v0.5.0 现网分发中）：按住热键说话，
松开后文字直接上屏到光标处，纯本地识别（sherpa-onnx + SenseVoice/Fun-ASR-Nano）。

本方案将其**全功能对等移植**到 **macOS arm64**。核心判断：识别、录音、模型下载、更新、
语义顺滑、按住/单击判定等主体资产天生跨平台；真正要重写的只有钩子、上屏、自启、
单实例、指示条、打包六块——且耦合已被现有 `winio`/`winutil` 门面隔离在 5 个文件里。

## 1. 决策记录（九项，均已拍板）

| # | 决策点 | 结论 | 关键理由 |
|---|---|---|---|
| 1 | 目标形态 | **全功能对等移植**，按里程碑推进 | 核心动作（按住说→上屏）做到 90 分即成立；不缩水 |
| 2 | 机型/系统 | **仅 arm64**，下限 **macOS 15** | 开发机 M5 原生闭环；Intel Mac 推理掉速无价值；下限被 sherpa-onnx 轮子击穿（见 §2.2），从 13 上调 |
| 3 | 代码架构 | **就地镜像 + 调度门面**：win 五模块原地不动，新增 mac 镜像模块；`app/native.py` 是全仓唯一 `sys.platform` 出现点 | Windows 现网产品零回归风险优先；mac↔win 模块同名函数面可机械化对照 |
| 4 | 默认热键 | **右 Cmd**（flagsChanged 观察） | 工程稳健优先：零压制、无 LED 失步、无重合成 |
| 5 | CapsLock | **mac 彻底不实现**；钩子层 listen-only tap | 贯彻稳健立场；tap 对系统零干预，回归面小一个量级 |
| 6 | 原生栈 | **pyobjc 统一**（Quartz/Cocoa/ServiceManagement 条件依赖）+ `tray.py` 双平台共用 pystray | pyobjc 反正被 pystray 拖进来；托盘是全功能对等的主对账单位，一份代码是刚需 |
| 7 | 权限引导 | **检测 + 深链 NSAlert + 诊断对话框常驻**；未授权静默降级无硬门 | mac 工具类标准姿势；诊断对话框是 Windows 已有交互的天然归宿 |
| 8 | 分发签名 | **ad-hoc 起步**：zip(onedir .app) 挂 Release；`--codesign-identity` 参数化留公证接口；bundle ID `com.xianglun918.xxl-whisper` 第一天固定 | 免费工具不为零收入预付 $99/年；bundle ID 固定则晚买零损失 |
| 9 | 回归验证 | **Windows 实机终验 + 双平台 CI 门禁**（ruff/basedpyright/pytest 非集成） | 门面化要动 6 个共享文件的 import 行，"Windows 零回归"必须被机械验证 |

## 2. 事实基础（两轮探查结论，一手验证）

### 2.1 代码耦合现状（全仓代码探查）

- **零平台分支**：`sys.platform`/`os.name`/`IS_WINDOWS` 全仓 0 处。5 个 Win32 模块
  （`winio`/`winutil`/`hotkey`/`mousehook`/`indicator`）import 时即 `ctypes.WinDLL`，
  在 mac 上第一个 import 就崩——耦合是结构性的，不是分支性的
- **耦合隔离良好**：`app.py`/`controls.py`/`tray.py`/`emit.py`/`main.py`/`update_flow.py`
  只消费 winio/winutil 的**函数表面**；`hotkey_logic.py` 的按住/单击判定是纯逻辑，
  天生为"钩子后端可替换"设计（最强接缝）
- **mac 上会消失的概念**：WM_PASTE / UIA 上屏通道、`keyboard_injection_alive` F13 探针、
  指示条 `WS_EX_NOACTIVATE`、注册表自启、内核互斥体单实例
- **mac 上无对应物的依赖**：`uiautomation`（uia.py 懒加载，且未声明在 pyproject）

### 2.2 依赖与打包事实（PyPI 轮子实测 / otool 实测 / 官方文档）

| 项 | 结论 |
|---|---|
| sherpa-onnx / core 1.13.7 | ✅ arm64 + x86_64 + universal2 轮子齐全（cp310–314），onnxruntime 已内置。**⚠️ 内置 ORT 强引用 CoreML `MLComputePlan`（macOS 15+ 才有）且 minos 26.x：在 macOS ≤ 14 上 dyld 加载期硬崩**，上游开放 issue [k2-fsa/sherpa-onnx#3840](https://github.com/k2-fsa/sherpa-onnx/issues/3840) 无修复时间表。15–25 上仅一条无害版本警告。⇒ 下限抬至 15（决策 #2） |
| pystray 0.19.5 | ✅ mac 后端纯 AppKit；`pyobjc-framework-Quartz(≥7.0); sys_platform=="darwin"` 条件依赖自动装，且 Quartz→Cocoa 传递链自动带出 AppKit。**限制：mac 无 radio 单选菜单**（`HAS_MENU_RADIO=False`，用 check 勾选替代，行为零差异）；`icon.run()` 主线程阻塞（与现架构"主线程跑托盘"同构） |
| sounddevice 0.5.6 | ✅ mac 轮子内置 **universal2 PortAudio dylib**（实测 x86_64+arm64 双架构），零系统依赖；CoreAudio 触发麦克风 TCC |
| PyInstaller 6.22 | ✅ `--windowed` 出 .app；**默认 ad-hoc 签名**（arm64 强制）；`BUNDLE(info_plist=…)` 注入自定义键；官方建议常驻托盘应用用 **onedir** 而非 onefile；非交叉编译，mac 上构建 mac 包 |
| Gatekeeper | 未公证应用网络分发被拦（"无法验证开发者"），**右键→打开**一次性绕过——与 Windows 版「杀软误报加白」同量级摩擦 |
| TCC 权限 | 麦克风：**必须** Info.plist 声明 `NSMicrophoneUsageDescription`，否则打包后静默拒绝；listen-only tap → **输入监控**；CGEventPost 注入 → **辅助功能**；三项均无法预授、无法绕过，必须做首启引导 |
| 微软官方 ORT vs sherpa 自建 | 微软版 MLComputePlan 为弱引用可跑 13/14；sherpa 自建为强引用——这是 #3840 的根因，也是不能靠"钉老版本"解决的依据（1.12.x 时代 libs 同样 15.5 minos，且 Fun-ASR-Nano 需要新版 sherpa-onnx） |

## 3. 移植蓝图

### 3.1 新增（mac 侧 + 基建）

| 文件 | 职责 |
|---|---|
| `app/native.py` | 调度门面：全仓唯一 `sys.platform` 点，win 下 re-export `winio`/`winutil`/`hotkey`/`mousehook`/`indicator`，darwin 下 re-export mac 镜像；并暴露数据根（win: `%LOCALAPPDATA%`，mac: `~/Library/Application Support/xxl-whisper/`） |
| `app/macio.py` | 镜像 `winio.py` 函数面：NSPasteboard 剪贴板读写/还原、CGEventPost 注入 Cmd+V、key_name（mac keycode→显示名）、active_monitor_work_area（NSScreen）、全屏检测（NSWorkspace 近似） |
| `app/macutil.py` | 镜像 `winutil.py`：NSAlert 对话框、SMAppService 自启、锁文件单实例、`osascript display notification` 通知、三权限检测（`AXIsProcessTrusted` 等）+ 深链 URL |
| `app/machotkey.py` | listen-only CGEventTap（右 Cmd / F 键 / 自定义捕获）+ CFRunLoop 专属线程 + 超时禁用看门狗。**永不压制、永不重合成**（决策 #5） |
| `app/mac_indicator.py` | 「正在听…」条：AppKit 非激活 NSPanel（`NonactivatingPanel` + floating level，保证不抢键盘焦点——上屏链路承重墙）；字体 PingFang SC |
| `build.sh` + `xxl-whisper-mac.spec` | onedir `.app` 打包；`info_plist` 注入 `NSMicrophoneUsageDescription` / `LSUIElement=true` / `CFBundleIdentifier=com.xianglun918.xxl-whisper`；`--codesign-identity` 参数化（空 = ad-hoc） |
| `.github/workflows/ci.yml` | 双平台门禁：`windows-latest` + `macos-latest`（arm64），`uv sync` → ruff → basedpyright → `pytest -m "not integration"`；只做门禁不做产物 |
| `scripts/probe_mac_*.py` | mac 探针组（M1 冒烟用，见 §4） |

### 3.2 修改点（机械性，Windows 行为零变化）

- **import 重指向**：`app.py` / `controls.py` / `tray.py` / `emit.py` / `main.py` / `update_flow.py`
  六文件 `from app import winio/winutil` → `from app import native`
- `emit.py`：通道阶梯改**按平台注册**——win 注册 4 通道（KEYS→WM_PASTE→UIA→CLIPBOARD），
  mac 注册 2 通道（KEYS-Cmd+V→CLIPBOARD）；阶梯编排逻辑共享
- `config.py`：`config_dir()` 数据根改走 `native`；mac 预置键名（`right_cmd`/`f2 f4 f6 f8`/
  自定义 mac keycode）；砍 `scroll_lock`/`mouse_x1/x2`（mac 无此键）
- `tray.py`：预置表按平台；radio→check（探测 `HAS_MENU_RADIO`）；预置标签经 `native.key_name`
- `pyproject.toml`：条件依赖 `pyobjc-framework-Quartz/Cocoa/ServiceManagement; sys_platform=="darwin"`；
  basedpyright 对 pyobjc 无类型表面沿用 ctypes 同款豁免先例
- `tests/test_winio_clipboard.py`：加平台守卫（mac 上收集期即崩）
- `README.md` + `docs/使用与分发说明.md`：mac 章节（安装/权限引导/已知边界/打包发版）

### 3.3 零改动资产（可移植核心，两 OS 共享）

`asr.py` / `recorder.py`（mic 名跨机不通用，mac 首启重选即可）/ `downloader.py` /
`updater.py` / `hotkey_logic.py` / `text_filter.py` / `build.bat` / 全部 `win*.py` 模块 /
现有 probe scripts。

## 4. 里程碑（全绿才进下一关）

- **M0 — 分支与基建**（✅ 完成）：切 `feat/macos-support`；本计划文档首 commit；
  `text_filter.py` 两件套落位；pytest `pythonpath` 修复（裸 `uv run pytest` 跨平台可解析）；
  平台测试守卫（win32-only 收集期跳过）；CI 双平台门禁（windows-latest + macos-15）
- **M1 — Day-1 冒烟探针**（✅ 4×OK，2026-09-10 本机验证）：
  ① `probe_mac_tap` ✅ listen-only @ `kCGSessionEventTap` 观察右 Cmd（keycode `0x36` flagsChanged），
  **看门狗 disable 静默 / enable 恢复双验证通过**；预检 API `CGPreflightListenEventAccess`
  ② `probe_mac_paste` ✅ 胜出注入配置 = **`kCGEventSourceStateCombinedSessionState` 源 →
  `kCGSessionEventTap` 投递点**（三配置全通，取最简）；序列 RCmd↓ → v↓ → v↑ → RCmd↑；
  剪贴板 300 ms 后还原验证通过。靶子 = TextEdit + **AX 回读**
  （`AXUIElementCopyAttributeValue(el, attr, None)` → `(err, value)`；树搜 `AXTextArea` 读 `AXValue`）
  ③ `probe_mac_permissions` ✅ 三权限检测 + 深链 URL；开发期授权归属宿主终端（iTerm2）
  ④ `probe_mac_tray` ✅ pystray-mac 可用；`HAS_MENU_RADIO=False` / `HAS_MENU_CHECK=True`；
  osascript 通知 exit 0
  ⚠️ **硬约束（CI 血泪教训）**：mac 模块的 pyobjc 导入必须走 `importlib.import_module` 动态化——
  静态 `import Quartz` 会让 windows runner 的 basedpyright 报 `reportMissingImports`，
  而加行级 `# pyright: ignore` 又会在 mac runner 报 `reportUnnecessaryTypeIgnoreComment`。
  先例：`app/uia.py` 的 `_auto()`
  ⚠️ **已知陷阱（勿重试）**：Tk 窗口吞合成 Cmd+V（即使 frontmost+key+focus 全满足）——mac 指示条
  必须用 AppKit NSPanel；`osascript 'make new document'` 会卡在 Automation 授权弹窗（故探针改用
  `open` + AX 回读，绕开 Automation）
- **M2 — 门面与重指向**（🚧 进行中）：`app/native.py`（全仓唯一 `sys.platform` 点）+ mac 骨架
  （macio/macutil/machotkey/macmousehook/mac_indicator/macuia）+ 八文件 import 重指向 + config 数据根；
  **Windows 全门禁必须仍绿**（CI windows runner + 实机）
- **M3 — 热键与录音链路**：`machotkey.py` 接入 → 源码跑通最小闭环
  「按住右 Cmd 说 → 识别 → Cmd+V 上屏」（macio 先只实现 paste 最小面）
- **M4 — macio 全函数面**：剪贴板还原（`restore_clipboard` 对等）、key_name、
  NSScreen 工作区、NSPanel 指示条、emit 双通道注册
- **M5 — macutil 与托盘全菜单对等**：SMAppService 自启、诊断对话框（含三权限状态 +
  深链按钮）、暂停/选麦/换热键/换模型/语义顺滑开关/更新检查逐项对照 pass
- **M6 — 打包与发版**：spec/build.sh → zip(onedir .app)；文档双平台化；
  Windows 实机终验（build.bat → exe → probe 三连 + 手测上屏链路）；
  Release 挂双附件（exe + arm64 zip）；发 v0.6.0

## 5. 已知对等偏差（全部进 README「已知边界」）

1. mac 默认键右 Cmd，**无 CapsLock 选项**（Windows 侧不受影响）
2. 上屏 2 通道 vs Windows 4 通道（mac 无 WM_PASTE/UIA 概念；Secure Input 密码框 → 剪贴板兜底）
3. 托盘单选菜单显示为勾选而非圆点（pystray-mac 能力限制，行为零差异）
4. 指示条全屏自动隐藏用 NSWorkspace 近似，与 Win32 独占全屏检测语义有差
5. 需 macOS 15+；15–25 首次加载 sherpa dylib 有一条无害版本警告
6. 首次打开需右键→打开（未公证；同 Windows「杀软误报加白」量级）
7. 首启需三项授权引导（输入监控/辅助功能/麦克风——Windows 完全没有的环节）
8. 产物形态分叉：mac 为 onedir .app 打 zip（Windows onefile exe 不变）
9. 开发期：源码运行时 TCC 授权归属宿主终端（README 写明，`tccutil` 可重置重测）

## 6. 风险与预案

| 风险 | 预案 |
|---|---|
| pystray-mac 菜单冒烟失败（M1-④） | 门面下换 rumps，受控替换点 |
| 右 Cmd 与个别用户习惯冲突 | 预置 f2–f8 + 自定义捕获天然覆盖 |
| sherpa-onnx #3840 上游修复 | 下限自动跟涨，零成本；不主动等 |
| tap 被系统超时禁用 | 看门狗定时 `CGEventTapEnable` 重启（M1 验证） |
| Windows import 层回归 | CI 双矩阵 + M2/M6 两次实机终验 |
| 老流程扰动 | `build.bat`/exe 管线/发布流程一行不改；uv.lock mac 轮子本就在 |

## 7. 验证策略

- **每 push**：CI 双平台门禁（ruff / basedpyright / pytest 非集成）
- **mac 实机**（本机 M5 / macOS 26）：M1 探针四连 + 每里程碑手动 pass（热键→录音→识别→
  上屏→还原→指示条→托盘全菜单）
- **Windows 实机**：M2、M6 两次终验（build.bat → exe → probe_focus/hold_bar/emit_chain
  三探针 + 手测上屏链路与托盘）
- **质量门禁标准**：与现网一致（`使用与分发说明` 的 ruff/basedpyright/pytest 全绿再发版）

## 8. 参考链接

- sherpa-onnx macOS 轮子与 #3840：<https://github.com/k2-fsa/sherpa-onnx/issues/3840>
- pystray 源（mac 后端）：<https://github.com/moses-palmer/pystray>
- PyInstaller macOS 文档（.app/签名/多架构）：<https://pyinstaller.org/en/stable/feature-notes.html>
- Apple 公证要求：<https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution>
- Apple 开发者计划（$99/年）：<https://developer.apple.com/support/enrollment/>
- Info.plist 键：`NSMicrophoneUsageDescription` / `LSUIElement`
  （<https://developer.apple.com/documentation/bundleresources/information-property-list>）
