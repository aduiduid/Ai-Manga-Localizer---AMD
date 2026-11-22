# 三种 OCR 后端横向对比（2026-11-22）

同一本英文漫画、同一套检测框、同一套翻译与渲染参数，只换识别器。
结论先行：**纯英文书用内置（Paddle）；Hayai 只对日文有增益；强制 Hayai 读英文会退化。**

## 对照对象

| 后端 | `ocr_backend` | 识别路径 | 任务记录 |
| --- | --- | --- | --- |
| 内置专用 OCR | `builtin` | Paddle 检测 + MangaOCR 识别（拉丁文不经注入 reader） | `a0751c788486` |
| Hayai 混合 | `hayai` | Paddle 检测 + Hayai 识别（日文页）/ Paddle 识别（拉丁页） | `df20180dea1c` |
| 仅 Hayai（强制） | `hayai_force` | Paddle 检测 + Hayai 识别**全部页面** | `601eccb646f6` |

书：`1416037-[Alcor] A Display of Passion (My Little Pony Friendship is Magic) [Digital] [English]`，74 页，全英文。

## 方法

- 三者共用 Paddle 检测，**1372 个 unit 的 id 与 bbox 完全一致**（逐 unit 校验，bbox 差异 0），因此差异只能来自识别器。
- 文本对齐键为 `(页文件名, unit id)`，比较前把空白归一化。
- 速度取任务记录 `result.timings`，不取墙钟：三者都在同一台机器、同一天跑，但翻译阶段有 GPU 竞争。
- 渲染图对比**用当前代码从各自缓存重渲**，不用目录里的历史产物——`已翻译` 目录的图产生于 0.13.15 之前的渲染器，直接比会把旧渲染 bug 算到 Paddle 头上。

## 一、速度

| 后端 | OCR (s) | 翻译 (s) | 渲染 (s) | 墙钟 (s) |
| --- | --- | --- | --- | --- |
| builtin | 1940.9 | 684.2 | 128.7 | 2760 |
| hayai | 1948.6 | 702.9 | 139.2 | 2793 |
| hayai_force | 2196.2 | 735.0 | 146.5 | 3078 |

- `builtin` 与 `hayai` 的 OCR 只差 7.7 s（0.4%）：**在英文书上两者识别路径相同**，见下节。
- `hayai_force` 慢 13%：全部英文页都走 CPU 上的 Hayai，而 Hayai 在英文上的速度优势不成立。
- `已翻译` 目录另有一条 1.7 s 的缓存命中记录（`0b440ea7404a`），那是复用了 OCR 缓存，**不是一次真实 OCR**，不能拿来比速度。

## 二、识别质量

### 三者归一化后完全一致的部分

`hayai` 与 `builtin` 的 **1372 个 unit 全部一致，差异 0 个**。这不是"质量相同"，而是**混合后端在英文页上一个字都没用自己的识别器**：`PaddleMangaOCR` 的 `_reads_latin` 会把拉丁文直接判给 Paddle 自己的识别器，注入的 reader 根本不参与。这正是 0.13.16 的既定行为。

### `hayai_force` 相对 Paddle 的分歧

929 / 1372 个 unit（67.7%）与 Paddle 不同。按可机械判定的失败模式归类：

| 模式 | 数量 | 占分歧 |
| --- | --- | --- |
| `differs`（长度相近，属实质差异） | 859 | 92.5% |
| `ellipsis`（整段塌成省略号） | 37 | 4.0% |
| `atrophy`（输出不足原文 45%） | 15 | 1.6% |
| `growth`（输出超原文 2.2 倍） | 11 | 1.2% |
| `runaway`（片段自我复读） | 7 | 0.8% |

**机械损坏合计 70 个（7.5%）；另有 859 个是普通误识**。

典型样本（原文 → 强制 Hayai）：

- `p046u13`：`AH! AH! AHH! AH AH! AH! GAH!` → `AHAHAHAHA…`（复读溢出，该 unit 因译文超长被丢弃，最终保留英文原文）
- `p044u03`：`YOU CAN'T HAVE YOUR…` 整段 → `MISCANY...`（塌成一句）
- `p069u03`：整页致谢名单 → `...`（37 个省略号之一；同页 `p069u04/u06/u07` 同样）
- `p073u02`：`This project was our first big debut…` → `This projects works, arrivatting also techniction…`（语义崩坏）
- `p063u04`：密集小字 `...QUESTRIA ARE WIDELY BELIEVED TO CROSSBREED...` → 输出更长但英文字母间距被拆散，译文只能压在原英文上重排

### 译文被丢弃的数量

| 后端 | 已翻译 unit | 保留原文（翻译失败/被弃） | 显式跳过 |
| --- | --- | --- | --- |
| builtin | 1324 | 13 | 48 |
| hayai | 1324 | 12 | 48 |
| hayai_force | **1286** | **51** | **86** |

强制 Hayai 多丢 38 个译文、多跳 38 个 unit。丢掉的都是上面那些 `ellipsis` / `runaway` / `atrophy` 的产物——翻译器拿到不是句子的输入，自然交不出译文。

## 三、渲染图

三份产出的同区域对照（当前代码重渲，同一 bbox 加同一边距）：

- **封面**：三者都保留 `Display of Passion` 与 `by Alcor` 原图，无底板、无擦除。三者在该区域与源图**逐像素相同**（0/905600 改动）。
- **长段对白 `p044u03`**：Paddle 与混合版译文完整落在气泡内；强制版整段缺失，只剩一句「混蛋……」，气泡大面积空着。
- **拟声词 `p046u13`**：Paddle 与混合版「啊！啊！啊啊！啊！好痛！」；强制版因译文超长被丢弃，**完全没有中文**。
- **密集小字 `p063u04`**：Paddle 与混合版排满且无溢出；强制版中文压在原英文上，右侧文字被气泡边界切断。
- **后记 `p073u02`**：Paddle 与混合版语义正确；强制版译文明显偏离原意。

结论：**渲染器本身在三者间无差别**，所有视觉差异都来自识别文本。

## 四、结论与配置矩阵

| 场景 | 推荐后端 | 理由 |
| --- | --- | --- |
| 纯英文漫画 | **`builtin`** | 与 `hayai` 完全等价（拉丁文都走 Paddle），但没有额外模型加载成本；`hayai_force` 会退化 |
| 日文漫画 | **`hayai`** | 前次实测：日文 3 页 27.1 s vs 36.8 s，且修掉了 MangaOCR 的整句乱码 |
| 日文 + 特殊字体 | `hayai_force` | 只有需要"连日文页也强制 Hayai"时才有意义，且要接受英文页退化 |
| 拉丁文的特殊字体 | `hayai_force` | 唯一能让 Hayai 读英文的方式，但**质量低于 Paddle**，只作兜底 |

**Hayai 在英文上的硬伤**：

1. **整段塌陷**：遇到低对比小字或紧排段落直接输出 `...`（37 例）或极短碎片（15 例）。
2. **复读**：拟声词/重复音节会无限展开成 `AHAHAHA…`（7 例），比原文长数倍。
3. **语义漂移**：正常长度的段落也会出现 `This projects works, arrivatting also techniction` 这类并非拼写错误、而是整句重构错误的输出。

这三点都不是"多认错几个字母"，而是**产不出可翻译的句子**，所以丢译率从 1.0%（13/1372）升到 3.7%（51/1372）。

## 复现

```
# 数据来源
data/jobs/601eccb646f6.json   # hayai_force -> 已翻译\单一
data/jobs/df20180dea1c.json   # hayai       -> 已翻译\混合
data/jobs/a0751c788486.json   # builtin     -> 已翻译

# 逐 unit 对比（bbox 一致性 + 归一化文本 + 分歧分类）
python _home/tmp/compare_three.py
python _home/tmp/quality_modes.py

# 渲染对照（用当前代码重渲，避免历史产物干扰）
python _home/tmp/rerender_book.py <书本目录> <源图目录> <输出目录>
python _home/tmp/build_units.py
```
