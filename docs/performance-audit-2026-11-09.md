# 性能体检报告（0.10.4，llama.cpp 全链路）

实测日期：2026-11-09。装置：`_home\tmp\prof1.py` / `prof2.py` / `prof3_batch.py` / `prof4_missing.py`，
对 `LlamaVisionOCR._chat` 与 `LlamaChatClient.chat/chat_json/chat_vision` 打桩计时，按阶段打点。
硬件：RX 9070 XT（Vulkan，`-ngl 99`）、Ryzen 7 5700X（16 线程）、`llama_context_tokens=65536`、`-np 1`。

测试页：
- `en_dense` = `1416037-[Alcor] A Display of Passion ... [English]\00000041.webp`（34 units）
- `en_densest` = 同目录 `00000061.webp`（36 units，全库最密）
- `ja_vertical` = `678390-(Kansai! Kemoket 2) ... solitary pupa ...\00000022.webp`

## 一、OCR 阶段：55%–62% 花在「逐 unit 的紧致裁剪重读」

`HybridMangaOCR.analyze` 六段耗时（秒，括号为占比）：

| 阶段 | en_dense | en_densest |
| --- | --- | --- |
| 模型导入（一次性） | 15.5 (5%) | 6.3 (1%) |
| Paddle 检测 + MangaOCR 初始化 | 46.0 (15%) | 40.5 (8%) |
| `refine`（整页语义判定） | 76.5 (24%) | 73.5 (14%) |
| `find_missing`（红框标注图 + 原图，2 图 1 次） | 65.9 (21%) | 78.1 (15%) |
| `ink_tile_hints` + `recover_missing` | 0.1 (0%) | 0.0 (0%) |
| **`refine_local_crops`（每 unit 一次裁剪调用）** | **109.9 (35%)** | **329.6 (62%)** |
| 合计 | 313.9 | 528.0 |

- VLM 调用次数：34 units 的页 **36 次**（refine 1 + find_missing 1 + 裁剪 34），36 units 的页 **38 次**。
- 裁剪调用平均 **9.2 秒/次**（329.6 s / 36 次），其中模型生成只有约 1.3 s（38 tok/s，输出约 50 token），
  其余是图像编码与单请求固定开销 —— **这就是为什么本报告的第一条建议是"合批"而不是"降 token"**。

## 二、翻译阶段：9B 首译 46% + 9B 终审 33%，Hy-MT2 只占 9%

两页（34 + 7 units）合计 **74.8 s / 21 次调用**：

| 阶段 | 秒 | 占比 | 调用数 | prompt 字符 |
| --- | --- | --- | --- | --- |
| 1. editor 9B 首译 | 34.7 | 46% | 13 | 16,698 |
| 2. Hy-MT2 1.8B 候选 | 6.7 | 9% | 5 | 5,471 |
| 3. 9B 终审 `review_pages` | 24.7 | 33% | 1 | 3,484 |
| 4. A/B `judge_candidates` | 5.4 | 7% | 1 | 310 |
| 5. 风险重译 `retranslate_risk_units` | 3.3 | 4% | 1 | 721 |

首译阶段内部：**2 次块调用**（20.0 s + 5.2 s，各 2,481 字符）
+ **11 次逐条补救**（`_translate_one`，每次 0.9–1.6 s，各重发约 1,354 字符）。
也就是说 34 个 unit 里有 11 个没通过 `_candidate_is_acceptable`，掉进了单条兜底路径，
**多付了 11 次完整 prompt 的代价**。

## 三、进程管理：启动与 KV 复用都健康，瓶颈不在这一层

- `load_model` 到 `listening` 用 **4.7 s**（模型 4.66 s）；进程级启动耗时 6.5–14.2 s（含 exe 与 DLL 冷启动）。
- 批次内由 `hold_llama_servers()` 持有，整个队列只在开头付一次启动、结尾停一次 —— 已是最优形态。
- 日志里 `graphs reused = 1520+`、`graph reallocs = 0` → **图与 KV 复用正常，没有反复重建**。
- `n_slots = 1, n_ctx_slot = 65536, kv_unified = 'false'`（`-np 1`）：所有请求**串行**，一次只有一个在途。
- 关键比例：**prompt eval 558 tok/s，生成 38 tok/s** —— 每个输出 token 比输入 token 贵约 **14.7 倍**。

## 四、五个候选提案（按"证据强度 × 收益"排序）

### P1 裁剪审计合批：`refine_local_crops(batch_size=4)` —— 实测 3.9×

同一页 36 个 audit unit，三种批量（`prof3_batch.py`）：

| batch | 耗时 | 调用数 | 相对 |
| --- | --- | --- | --- |
| 1（现状） | 327.7 s | 36 | 1.00× |
| 2 | 92.9 s | 18 | 3.53× |
| 4 | 84.0 s | 9 | 3.90× |

质量对照（逐 unit 签名 diff，batch=1 为基准）：36 个里只有 2 个不同 ——
`p001u04` 由 `sfx:impact` 变 `semantic_role_conflict`（`SNRK`，仍是 sfx）；
`p001u10` **变好**：单条时把西语笑声读成日文乱码，合批后读出正确的
`HAHAH HAHAHAH! HAHALL HAHAHAH!`。

- 现状：`batch_size` 参数早已存在（`ocr.py:2678`），但唯一调用点 `ocr.py:3228` 从未传参。
- 收益：en_densest 的 OCR 从 528 s → 约 283 s（**1.87×**）；按 74 页的书折算，OCR 阶段省掉一半以上。
- 风险：低。测试里 11 处调用都不传 `batch_size`，保持默认 1 不受影响。
- 验收：英文 5 页 + 日文 1 页逐 unit A/B；输出图逐像素 diff。

### P2 英文书直接跳过 `find_missing` 与 `ink_tile` —— 可证明的输出不变，白捡 15%–21%

证据链：`find_missing` / `ink_tile_hints` 的产出**唯一**去向是 `recover_missing`
（`ocr.py:3206`），而 `recover_missing`（`ocr.py:1736`）第一行就是
`if self._declared_language() == "en": return page`。
所以声明英文时这两步的提案**永远无法变成几何**，却实打实花掉 65.9 / 78.1 秒。

- 收益：英文页 OCR 直接减掉 15%–21%（配合 P1，en_densest 约 528 → 205 s，**2.6×**）。
- 唯一副作用：`.ocr.json` 的 `semantic_missing` 从"N 条 unconfirmed"变成空数组（纯记录字段，不参与翻译与渲染）。
- 风险：极低（可证明成图逐字节相同）。需要一条单测锁住"英文路径不再调用 find_missing"。

### P3 治理首译的「逐条回退」（先测量后改）

11 次单条补救每次重发 1,354 字符 prompt。若能把块级一次成功率提上去，
阶段 1 有望从 34.7 s 降到 25 s 左右。

- 第一步（不改行为）：在 `_candidate_is_acceptable` 的四个拒绝分支打点，统计真实原因分布。
- 第二步：按主导原因对症（例如长度下限、SFX 白名单、标点规范），或把被拒行聚成一个小批次重译，
  而不是逐条发。
- 风险：中。动的是翻译契约，必须严守"不许降阈值掩盖"的原则。

### P4 终审按风险分级（质量敏感，建议最后做或做成选项）

`review_pages` 单次就是 24.7 s（33%），且它对**所有**可译行都送审。
`SEMANTIC_RISK_RE`（`translator.py:46`）已经筛出了语义高风险行。

- 方案：只有命中风险正则、或两候选分歧、或首译被修过的行才进 9B 终审；其余靠 Hy-MT2 候选做确定性比对。
- 收益：终审耗时随风险行比例下降（密集合本里风险行通常不到三成）。
- 风险：**高**。这是唯一直接动输出质量的提案，建议排在 P1/P2 之后单独 A/B。

### P5 并发与 KV 前缀复用（实验性，收益待证）

两个独立方向，都要实测：

1. **`-np 2`**：单槽导致全程串行，而每次裁剪调用有约 9 s 的固定开销。
   两槽在途可让一次请求的生成与另一次的图像编码重叠。代价：`-np 2` 会把 65,536 上下文对半分
   （32,768/槽，仍远大于实际用量）。
2. **图像挪到消息末尾**：`chat_vision` 现在把 image parts 放在指令**之前**（有测试锁定），
   于是每次请求的前缀都从图像开始、**必然全量重算 prompt**。
   把固定指令放前面、图像放后面，可让 llama.cpp 复用相同前缀的 KV（每次省一次 prompt eval）。

- 风险：中；收益不确定（受 llama.cpp 多模态前缀缓存实现限制）。建议作为最后一个实验，不做承诺。

## 五、建议的落地顺序

1. **P1 + P2**（当天可完成，互不冲突，合计把英文密集页 OCR 从 528 s 压到约 205 s）；
2. 跑 P3 的诊断打点（不改行为），拿到拒绝原因分布再决定改法；
3. P5-1（`-np 2`）单独实测；
4. P4 与 P5-2 按需，且必须单独 A/B。

每一项的验收口径一致：日文页与英文页各 A/B 一次（unit 数、逐 unit 文本/坐标、成图像素 diff、
`quality` 字典），外加全量 `pytest`。
