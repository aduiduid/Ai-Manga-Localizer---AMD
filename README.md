# Manga Localizer Studio · AMD 显卡个人适配版

本地优先的漫画中文化工作台：按自然页序检测文字、识别原文（日文／英文）、结合前后页保持剧情与称呼连贯，最后只在文字区域内重绘中文 —— **不覆盖源文件**，也不把整页图片丢给云端。

本仓库是 [whmc76/manga-localizer-studio](https://github.com/whmc76/manga-localizer-studio) **v0.5.2** 的个人分支：在上游基础上做了 **AMD 显卡（Radeon RX 9070 XT）+ Windows** 适配，并补齐了安全暂停、任务计时、英文源支持、模型页真实状态等一批本地改动。相对上游的逐条改动、踩坑与实测数据见 [`README-本地部署.md`](README-本地部署.md)；上游的设计基线见 [`docs/DESIGN_CONTRACT.md`](docs/DESIGN_CONTRACT.md) 与 [`docs/PARITY_LEDGER.md`](docs/PARITY_LEDGER.md)。

![桌面端工作台](docs/screenshots/app-desktop-local.png)

## 实机效果：原图与简体中文输出对比

下面三张图来自本机对 `[Alcor] A Display of Passion`（英文同人，My Little Pony）的真实运行：内置 OCR（PP-OCRv5 + Manga OCR）+ 本机 Ollama 9B 质量模式 + LaMa 补画 + 样式重建；没有人工人名表，也没有调用在线 LLM API。5 页共 30 个文字单元：替换中文 30 个、明确跳过 0 个、未解决 0 个、无效译文 0 个。

![第 3 页（整页后记文字）原图与简体中文输出对比](docs/screenshots/local/comparison-page-03.webp)

![第 4 页（对白密集页）原图与简体中文输出对比](docs/screenshots/local/comparison-page-04.webp)

![第 5 页（多气泡与效果字）原图与简体中文输出对比](docs/screenshots/local/comparison-page-05.webp)

每张图左半是原图、右半是输出；README 只做等比例缩略展示，成品保持源图原始像素尺寸（800×1132），擦除与补画只发生在检测器确认的文字区域内。

## 本分支相对上游做了什么

- **AMD/Windows 适配**：不依赖 CUDA。设备选择在任何 `gpu*`/`cuda` 请求上先做可用性检查并回落到 `cpu`（Paddle、LaMa、Hy-MT2 三处），否则 AMD 机器上会死在 oneDNN 的 PIR 转换里、或抛 `Torch not compiled with CUDA enabled`。
- **翻译全部上显卡**：把同一份 Hy-MT2 1.8B 权重交给本机 Ollama（Vulkan）跑质量模式的第二遍候选，`MLS_HY_MT2_OLLAMA` 一行开关。单次候选调用从 CPU torch 的 9–74 s 降到 0.07–0.09 s；整本 71 页质量模式从 **22.5 min 降到 4.3 min**。
- **英文源支持**：原文语言自适应（`日文`／`英文`）的 prompt、拉丁页剔除检测器的无文本画面框、拉丁音效表与双写拟声词识别、横排字号下限放宽。修复前英文气泡会被 Manga OCR 读成假名幻觉、译文全错。
- **「已可接受的译文是终稿」守卫**：审校／重译轮只能救援被硬门槛拒绝的行，不再把编辑初译的好译文覆盖成更差的版本。
- **安全暂停**：页与页、chunk 与 chunk 之间暂停（模型与已完成页面都保留），暂停时段不计入计时；配合原有的「结束」与设置页「退出程序」。
- **更完整的 UI**：侧边栏任务状态卡（阶段／页数／开始时间／已用时间／预计还需）、页码滑动条、任务记录回填、模型管理页显示**实际在跑**的模型（含两条 Ollama tag 与真实大小）、三语言文案补齐。
- **输出目录自动推导**为 `{输出根}\{源目录名}--已汉化`，中间产物与成品都落在该子目录。
- **纯 ASCII 安装路径**的要求与本地启动脚本（见下）。

## 特性

- **中日英界面**：工作台可即时切换中文、日文和英文，并自动记住选择；界面语言与漫画目标语言相互独立。
- **故事连贯**：默认携带前 6 页成稿；扫描全书反复出现的片假名角色名，通过 JMnedict、中文姓名候选过滤和本地 9B 裁决生成一致性约束，不需要为具体作品维护人工人名表。
- **本地质量集成**：质量模式由同一个 9B 模型分阶段完成草稿、上下文裁决和语义风险复译，Hy-MT2 1.8B 提供独立候选；不下载或依赖 9B 以上模型。快速模式仍可只运行单一轻量模型。
- **质量 OCR**：Paddle 精确框 + Manga OCR 识别 + 9B 漏字／语义审计；视觉模型报告的漏字必须经过已有框覆盖判断、局部放大 Paddle 重检和 Manga OCR 独立确认，粗 VLM 框永远不直接擦图。
- **原字样式匹配**：不依赖多模态 LLM，直接分析横竖排、字号、字重、前景色、描边色／宽度、字距和展示文字占比；自动使用常规或粗体中文字体重排。
- **保护画面**：保持原始像素尺寸，不覆盖源图；LaMa 推理后强制逐像素恢复掩膜外区域，回归测试验证有界编辑契约。
- **本地后端可组合**：翻译与 OCR 都可选 **内置**、**本机 Ollama**、**本机 llama.cpp**（0.10.0 起，直接读 Ollama 已下载的 GGUF）或显式在线兼容 API；Ollama 路径完整保留，随时可作 fallback 切回。
- **隐私边界清晰**：本地模式不外传数据；在线模式只发送 OCR 文本和上下文，不上传图片。

## 环境要求

| 项目 | 本机实测 |
| --- | --- |
| 系统 | Windows 10/11（Linux 亦可，脚本在 `scripts/`） |
| CPU | AMD Ryzen 5 5600G（6 核 12 线程，OCR 纯 CPU 跑 5–6 核） |
| 显卡 | AMD Radeon RX 9070 XT —— 只给 Ollama / llama.cpp 用 Vulkan，Paddle 没有 ROCm/DirectML 后端 |
| 磁盘 | 完整本地质量模式约 15 GB（含 DeepSeek-V4-Pro-Qwen3.5-9B i1-Q6_K 8.3 GB + Hy-MT2 1.9 GB 两份权重；llama.cpp 用硬链接复用同一批 GGUF，不额外占盘） |
| Python | 3.12.15（uv 托管），torch 2.8.0+cpu / paddlepaddle 3.3.1 / paddleocr 3.7.0 / paddlex 3.7.2 |
| 推理服务 | Ollama 0.40.14（免安装版放在 `_dl\ollama\`，只保留 vulkan 后端）；llama.cpp 后端直接复用其中自带的 `llama-server.exe`（`0.8.0-dev`，commit `39febd2ad`） |

**安装路径必须是纯 ASCII。** Paddle 的 C++ 推理和 `torch.jit.load` 都打不开含中文的路径，会报 `json.exception.parse_error.101` / `open file failed because of errno 2`。目录名请保持 `Manga Localizer----Ben-Ding-Fan-Yi` 这类纯 ASCII 形式，**不要再改成中文名**；如果搬动目录，还要同步改 `.venv\pyvenv.cfg` 的 `home` 和 `site-packages\_editable_impl_manga_localizer_studio.pth`。

## 安装与启动

**全新安装**：克隆后先跑引导脚本（`-Profile cpu` 会装 `torch==2.8.0+cpu`、`torchvision==0.23.0+cpu` 并卸载 `paddlepaddle-gpu`），再启动 UI —— 权重留给 UI 里的「准备缺失模型」按需拉取，省掉一次几 GB 的下载。

```powershell
git clone https://github.com/aduiduid/Ai-Manga-Localizer---AMD.git
cd Ai-Manga-Localizer---AMD
.\scripts\bootstrap.ps1 -Profile cpu -SkipModels
.\start-windows.bat
```

> AMD 机器上**不要装 CUDA 版 torch**：本分支的设备回落逻辑本身就假设 `torch.cuda.is_available()` 为假，CUDA 轮子既用不到又会让 `torch` 体积翻几倍。`bootstrap.ps1` 的 `-Profile auto` 只在检测到 `nvidia-smi` 时选 `cuda129`，其余情况回落 `cpu`；AMD 机器请显式写 `-Profile cpu`。若本机没装 `uv`，脚本会自动回落到 `python -m venv` + `pip` 的兼容路径。

**本机的启动方式**：本机把项目放在 `<根>\app\` 下（仓库根 = 项目根，环境在 `<根>\app\.venv`），根目录的 `start.bat` 已固化本机用到的全部环境变量 —— `MLS_HOME`、`PADDLE_PDX_CACHE_HOME`、`HF_ENDPOINT=https://hf-mirror.com`、`OLLAMA_MODELS`、`MLS_HY_MT2_OLLAMA`，并把 `USERPROFILE`／`TEMP` 挪到 `_home\`（Paddle 会往 `~/.cache/paddle` 写），同时自动拉起 `_dl\ollama\ollama.exe serve`。

```
start.bat                              :: 打开 http://127.0.0.1:8765
start.bat run "D:\书\某本" -o "D:\书\某本--已汉化"
start.bat doctor                       :: 环境 + 实际模型清单
start.bat models status
```

`start.bat` 假定 `%~dp0app\.venv\Scripts\python.exe` 已存在（CPU 版 torch）；换机器请改脚本里的 `PY` 路径。**改这个脚本必须保持 CRLF 行尾**，写成 LF 会被 cmd 拆碎成 `'Localizer' is not recognized...`。

## 使用

1. 右上角切换界面语言，选择漫画图片文件夹（JPG、PNG、WebP、BMP、TIFF）。
2. 输出目录会自动填成 `{输出根}\{源目录名}--已汉化`；换成别的目录也可以，但输出不能与源目录相同。
3. 选择 OCR 与翻译后端，按需设置「故事连贯」「前文页数」「保留拟声词」和输出格式（默认无损 WebP）。
4. 模型未就绪时点「准备缺失模型」；只会下载当前后端需要的权重。
5. 运行中可以**暂停**（当前页跑完停住，模型与已完成页面全保留）、**结束**（下一次进度回调停下，缓存复用）或从设置页**退出程序**。侧边栏卡片显示开始时间、已用时间与预计剩余。

CLI 与 UI 使用同一条流水线：

```bash
manga-localizer run "D:/manga/chapter-01" -o "D:/manga/chapter-01--已汉化"
manga-localizer run "D:/manga/chapter-01" -o "D:/manga/chapter-01--已汉化" --backend ollama --quality-profile quality
manga-localizer run "D:/manga/chapter-01" -o "D:/manga/chapter-01-fast" --backend ollama --quality-profile fast
manga-localizer run "D:/manga/chapter-01" -o "D:/manga/chapter-01-png" --output-format png
manga-localizer run "D:/manga/chapter-01" -o "D:/manga/chapter-01-final" --reviewed-transcript "D:/manga/transcript-reviewed.json"
manga-localizer models status
manga-localizer doctor
```

在线兼容 API 使用 `--backend online --online-url ... --online-model ...`，密钥通过 `MLS_ONLINE_API_KEY` 环境变量提供；UI 中输入的密钥只保留在当前服务进程内，不写入磁盘。

## 推理后端：Ollama 与 llama.cpp（0.10.0 起）

**两条全本地推理路径并存，Ollama 路径完整保留**，随时可以切回去：

| | Ollama | llama.cpp |
| --- | --- | --- |
| 服务 | 常驻守护进程（127.0.0.1:11434） | 按需起 `llama-server.exe`，任务/批次结束即 kill 并交还显存 |
| 权重 | `ollama-models\blobs\` 里的 GGUF | 硬链接复用**同一批 blob**，不复制、不转换 |
| 翻译 | `inference_backend=ollama` | `inference_backend=llama` |
| OCR | `ocr_backend=ollama`／`hybrid` | `ocr_backend=llama`（与 hybrid 同一套「几何＋语义」流程，只换视觉传输） |

**怎么配**：在项目面板或设置页把「推理后端」选成 `llama.cpp 本地推理`，「OCR 后端」选 `llama.cpp 视觉 OCR`，模型名沿用 Ollama 里那两个 tag（默认已填好）。

| 设置 | 默认值 | 说明 |
| --- | --- | --- |
| 主模型 | `DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q6_K:latest` | 同一份多模态 9B 供翻译与视觉 OCR 共用 |
| 审校模型 | `hy-mt2-1.8b:q8` | 质量模式的独立候选译文 |
| 上下文 | `65536` | 需要时可在 4096–262144 之间调 |
| 端口 | 主 `8080` / 审校 `8081` / 视觉 OCR `8082` | 被占用时自动顺延并在任务日志里说明；整段被占则明确报错 |
| llama-server 路径、Ollama 模型目录 | 留空 | 留空时自动在 `_dl\ollama\lib\ollama\` 与 `ollama-models\` 下查找 |

启动参数由程序拼装（`-ngl 99 -fa on -rea off -np 1 -ctk q8_0 -ctv q8_0 --no-webui`），并先在 `data\llama-models\runtime\` 里备好一份私有运行时：`llama-server.exe` 与 `ggml-vulkan.dll` 必须同目录，否则 llama.cpp 找不到 GPU 会静默回落到 CPU。视觉 OCR 要求模型带 mmproj 投影器（缺失时启动前就报错）；同 tag 的模型会被复用，不会为 OCR 再加载一份 7 GB 权重。页面在送进视觉模型前会转成 PNG（llama.cpp 拒收 WebP）。

CLI 与 UI 走同一条流水线，`--backend llama`／`--ocr-backend llama` 会沿用你保存的设置，单个 `--llama-*` 参数可覆盖本次运行：

```bash
manga-localizer run "D:/manga/chapter-01" -o "D:/manga/chapter-01--已汉化" --backend llama --ocr-backend llama
manga-localizer run "D:/manga/chapter-01" -o "D:/manga/chapter-01-llama8083" --backend llama --llama-main-port 8083
```

本机实测（同一页、质量模式）：llama.cpp 每页 43–45 s 且稳定，Ollama 同页会偶发跑到 333 s；启动 6.55 s，34/34 层全部 offload，KV 用 q8_0。llama.cpp 视觉 OCR 与 Ollama 的整页结果在 hybrid 流程下 **坐标 4/4、文本 4/4 完全一致**。完整对比见 [`CHANGELOG.md`](CHANGELOG.md) 的 0.10.0 一节。

## 工作原理

```mermaid
flowchart LR
  A["漫画文件夹\n自然页序"] --> B{"OCR 后端"}
  B --> B1["PaddleOCR + Manga OCR"]
  B --> B2["本机 Ollama 视觉模型"]
  B1 --> C
  B2 --> C
  C["精确几何＋整页语义复核"] --> D{"翻译后端 / 质量档"}
  D --> D1["快速：Hy-MT2 或单一 Ollama"]
  D --> D2["质量：同一 9B 分阶段审校＋Hy-MT2 候选"]
  D --> D3["显式可选：在线兼容 API"]
  D1 --> E
  D2 --> E
  D3 --> E
  E["LaMa 有界补画 + 原字样式分析\n粗体/描边/横竖排重建"]
  E --> F["无损 WebP/PNG + OCR/翻译稿\n独立输出目录"]
```

| 模块 | 默认方案 | 下载策略 |
|---|---|---|
| 文字定位 | PP-OCRv5 mobile det + server rec（CPU） | PaddleX 强制 `modelscope` 源 |
| 文字识别 | Manga OCR（日文）；英文页走 Paddle 识别结果 | 上游源回退，并在 UI 标明 |
| 连贯翻译 | 同一 9B 分阶段草稿／审校 + Hy-MT2 独立候选（均走 Ollama/Vulkan） | Ollama 自动拉取；不使用 9B 以上模型 |
| 画面修复 | Big-LaMa（质量）/ OpenCV（快速） | 约 196 MB，SHA-256 校验、原子下载 |
| 样式与排字 | OpenCV 分析 + Noto CJK Regular/Bold + Pillow | 不调用 LLM，不联网 |

模型与缓存都在项目目录内（`data\models`、`ollama-models\`），也可以用 `MLS_HOME` 改变位置；字体默认取系统 Noto Sans CJK SC，可用 `MLS_FONT` 指向自己的 TTF/OTF/TTC。

## 已知限制

- **OCR 是纯 CPU**：RX 9070 XT 只能给 Ollama 用 Vulkan，Paddle 没有 ROCm/DirectML 后端。整本 71 页质量模式实测约 3.6 s/页。
- **没有气泡、直接画在背景上的小字可能漏检**：把 PaddleOCR 短边放大到 1280 能读出来，但整页识别慢 32% 并引入噪声框，因此没有默认开启。
- 专有名词（书名、人名）单次翻译容易误译；质量模式已启用候选 + 9B 审校，仍不保证。
- 白描边小字擦除后可能留淡残影，改用 `quality_profile=fast` 或手动框可以规避。
- `preserve_sfx=True` 时英文音效按原样保留（`Grrrrr`、`poof` 等），不会译成中文拟声词。

## 设计与工程约束

- 检测、补画和排字规则只使用图像／文字单元特征，不包含作品标题、对白、页号、文件名或固定坐标。
- 原始图片从不被覆盖；API 也拒绝相同的源／输出目录。
- 像素变化限制在检测框周围的有界清理区；边缘连通的角色、背景和分镜线会被保留，回归测试覆盖该约束。
- 翻译保留稳定单元 ID，模型漏行或返回原文时逐条补译，避免气泡错位。
- 审校稿中的跳过项必须使用 `duplicate`、`noise`、`decorative` 或 `preserve` 之一作为 `skip_reason`；缺失原因的旧稿会被安全地视为未完成。
- 模型状态、进度、历史和预览均来自真实本地 API，不使用伪造 UI 状态。

## 开发与测试

```powershell
uv lock --check
uv run --frozen --no-sync pytest
uv run --frozen --no-sync manga-localizer ui --no-open
uv run python scripts/verify_output.py SOURCE OUTPUT OUTPUT/.manga-localizer-work/transcript.json
```

本机实测 **217 个测试收集、216 个通过**。唯一未通过的是 `tests/test_inpainting.py::test_lama_wrapper_preserves_every_unmasked_pixel`：`torch.jit.save` 打不开本机中文临时目录（pytest 的 `tmp_path` 落在中文路径下），属环境限制而非代码问题：

```powershell
python -m pytest tests --deselect tests/test_inpainting.py::test_lama_wrapper_preserves_every_unmasked_pixel
```

## 许可证与模型说明

项目源码使用 [MIT License](LICENSE)。下载的模型权重不随仓库分发，沿用各自上游许可证；详见 [MODEL_LICENSES.md](MODEL_LICENSES.md)。请只处理你有权翻译和发布的内容。

## 致谢

上游项目 [Manga Localizer Studio](https://github.com/whmc76/manga-localizer-studio) 由 **赛博迪克朗**（[@whmc76](https://github.com/whmc76)）开发并维护，本分支的 AMD／中文适配建立在 v0.5.2 之上。

核心能力来自 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)、[Manga OCR](https://github.com/kha-white/manga-ocr)、[LaMa](https://github.com/advimman/lama)、[ModelScope](https://modelscope.cn/) 与 [Hy-MT2](https://modelscope.cn/collections/Tencent-Hunyuan/Hy-MT2)。
