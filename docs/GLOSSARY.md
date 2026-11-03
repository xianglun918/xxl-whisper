# 术语表（产品语言 ↔ 代码标识）

这份表存在的理由：同一件事在代码、托盘菜单、配置文件和日志里各有一个名字，其中
几组已经漂过（`bilingual` 同时指"打字打两行"和"字幕画两行"；"上屏"既被用来指
"显示字幕条"，又被用来指"把文字打出去"）。改代码或改文案之前先在这里对一次词。

## 听写方式

| 中文（用户看到） | 代码 | 含义 |
|---|---|---|
| 按住说话 | push-to-talk / `_holding` | 按住热键期间录音，松手即一句，不需要 VAD |
| 持续听写 | continuous / `ContinuousSession` | 常开麦克风，靠 VAD 切句；托盘开关 |

## 输出去向

| 中文 | 代码 | 含义 |
|---|---|---|
| 上屏（字幕条） | overlay / `OverlayModel` | 屏幕上的半透明字幕窗口，**不**注入按键 |
| 打字 | typing / `EmitQueue` + `TypeBatcher` | 把文字注入到当前焦点控件（剪贴板粘贴） |
| 输出方式 | `overlay_output` = `screen` / `both` / `type` | 三选一：只上屏 / 上屏+打字 / 只打字 |
| 字幕布局 | `overlay_mode` = `target` / `source` / `stacked` | 字幕条画几行：只译文 / 只原文 / 原文在上译文在下 |

> `stacked` 就是 v2 里的 `bilingual`。改名理由：`TranslateMode.bilingual`（打字
> 时原文+译文各一行）和它**不是一回事**，两个 `bilingual` 指的是不同维度。
> 配置文件里旧值仍能读，迁移时会重写成 `stacked`（`CONFIG_VERSION = 3`）。

## 翻译

| 中文 | 代码 | 含义 |
|---|---|---|
| 翻译方向 / 线路 | route / `app/routes.py::ROUTES` | 中→英、英→中、日→中（直译）、日→英 |
| 直译单模型 | `models=("ja-zh",)` | 一个 145MB 模型直接出结果（日→中 的正式路径） |
| 两段回退 | `fallback=("ja-en","en-zh")` | 直译模型下不下来时的兜底链路，**不是**默认路径 |
| 整句翻译 | `translate()` | 一次端点一句，不做逐词流式 |
| 语义顺滑 | `disfluency` = `verbatim` / `smooth` | 只对 Fun-ASR-Nano 生效（换解码提示词），其余模型无效果 |

## 模型

| 中文 | 代码 | 含义 |
|---|---|---|
| 识别模型 | `model` / `ModelKind` | `sensevoice` / `whisper_turbo` / `funasr_nano` |
| 模型文件布局 | `app/model_files.py::MODEL_LAYOUT` | kind → {逻辑名 → 相对路径}；识别器只认逻辑名（`model` / `tokens` / `encoder`…） |
| 语音活动检测 | VAD / `Segmenter` | Silero VAD，负责切句与端点判定 |

## 交付与排障

| 中文 | 代码 | 含义 |
|---|---|---|
| 发布标签 | tag `vX.Y.Z` / `-` 后缀为预发布 | CI 按 tag 出三平台产物 |
| 恒定日志前缀 | `startup:` / `worker:` / `asr:` / `recorder:` / `hold:` / `press:` / `click:` / `loopback:` / `emit:` / `translate:` / `tray:` / `overlay:` / `config:` / `update:` / `continuous:` / 关闭时那条 `%d finished sentence(s) still queued` | 排障时按前缀 grep；措辞改动等于破坏排障资产（模型相关没有 `model:` 前缀，是 `model ready:`） |
