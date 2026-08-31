# xxl-whisper 落地页设计系统

> 来源：Layer B `vercel.md`（黑白精密 · Geist · shadow-as-border）+ Layer A `minimalist-skill.md`（克制 · 无 emoji · 无重阴影 · 柔和点彩）。
> 定位：开发者工具的现代落地页。单文件静态 HTML，无构建。

## 1. 氛围（Atmosphere）

画廊般的克制与精密——纯白画布、近乎黑（`#171717`）的文字，让每个元素都为结构服务。
“按住 CapsLock 说话”是唯一的签名瞬间：一个终端式演示卡，内含被按下的 CapsLock 键帽、
跳动的声音波形、以及逐字落下的中文文本。签名材料是 **shadow-as-border**（用 0 偏移 1px 阴影
代替传统边框），配合极轻微的径向光晕营造大气层，绝不使用渐变背景或重投影。

## 2. 色彩（Color tokens）

| Token | 值 | 用途 |
|---|---|---|
| `--bg` | `#ffffff` | 页面画布 |
| `--ink` | `#171717` | 标题、主文字（非纯黑，微暖）|
| `--ink-2` | `#4d4d4d` | 正文 |
| `--ink-3` | `#666666` | 辅助 |
| `--line` | `#ebebeb` | 分隔线 |
| `--line-soft` | `rgba(0,0,0,0.08)` | shadow-as-border |
| `--accent` | `#0a72ef` | 功能强调：链接、焦点环、主 CTA 悬停 |
| `--rec` | `#ff5b4f` | 语义色：录音状态点（仅演示卡与状态）|
| `--tag-blue-bg` / `--tag-blue-tx` | `#ebf5ff` / `#0068d6` | 蓝色药丸徽章 |
| `--tag-green-bg` / `--tag-green-tx` | `#edf3ec` / `#346538` | 绿色药丸徽章 |

规则：色彩是功能性的，从不用作装饰。单一 `--accent` 承担所有交互强调。

## 3. 字体（Typography）

- 展示/正文：`"Geist", -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif`
- 等宽（技术标签/按键/代码）：`"Geist Mono", ui-monospace, "SF Mono", "Consolas", monospace`
- 通过 Google Fonts 渐进增强加载 Geist / Geist Mono（`display=swap`）；被墙时回退系统栈，
  中文由 PingFang SC / Microsoft YaHei 渲染。
- 权重只用三档：400（正文）、500（交互）、600（标题），永不用 700。
- 展示标题 `letter-spacing: -0.02em`，行高 1.1；正文行高 1.6。
- 等宽标签大写 + `letter-spacing: 0.08em`，作为“开发者控制台”的声音。

## 4. 间距（Spacing）

8px 基座：`4 / 8 / 12 / 16 / 24 / 32 / 48 / 64 / 96 / 120`。区块垂直留白 96–120px，
内容最大宽度 1120px。宏留白即设计。

## 5. 组件（Components）

- **按钮**：主 CTA 实心 `#171717`/白字，圆角 6px，padding 12px 24px，`:active` 微缩 0.98；
  幽灵按钮白底 + `box-shadow: 0 0 0 1px var(--line-soft)`。
- **卡片**：白底，无传统 border，`box-shadow: 0 0 0 1px rgba(0,0,0,0.08), 0 2px 2px rgba(0,0,0,0.04), 0 0 0 1px #fafafa inset`；圆角 8px。
- **药丸徽章**：9999px，`--tag-*-bg`/`--tag-*-tx`，12px/500，仅作状态标签。
- **kbd 按键**：`border: 1px solid #ebebeb; border-radius: 4px; background:#f7f6f3`，等宽字体。
- **图标**：内联 SVG，1.75px 描边，无 emoji。

## 6. 动效（Motion）

只动 `transform` / `opacity`。滚动进入：`translateY(12px)+opacity:0 → 0`，600ms
`cubic-bezier(0.16,1,0.3,1)`，IntersectionObserver 触发，网格项级联 `calc(var(--i)*80ms)`。
波形条：`scaleY` 循环（代表“正在听”状态，有信息意义）。悬停：卡片阴影轻微加深；按钮 `:active scale(0.98)`。
无装饰性动画。

## 7. 深度（Depth）

| 层 | 处理 |
|---|---|
| 0 平 | 无阴影：画布、正文 |
| 1 环 | `0 0 0 1px rgba(0,0,0,0.08)`：卡片/分隔 |
| 2 卡 | 环 + `0 2px 2px rgba(0,0,0,0.04)` + 内层 `#fafafa` 环：重点卡片 |
| 大气 | 主视觉背后 `radial-gradient` 极淡光晕（opacity ~0.05）+ 细网格 |
