# Manga Localizer Studio 本地部署说明

上游：https://github.com/whmc76/manga-localizer-studio （v0.5.2，MIT）

安装在 `D:\Manga Localizer----Ben-Ding-Fan-Yi`（纯 ASCII 路径，无需任何 junction）。

## 怎么用

双击 `start.bat` → 浏览器打开 http://127.0.0.1:8765

命令行也可以，参数透传给 CLI：

```
start.bat run "E:\工作区\漫画\日文\某本" -o "E:\工作区\漫画\中文\某本"
start.bat doctor
start.bat models status
```

默认设置（`data\settings.json`）：OCR = PaddleOCR+Manga OCR，翻译 = 本机 Ollama 9B，质量优先，输出无损 WebP。

侧边栏左下角有任务状态栏：圆点颜色 + 当前阶段/页数，运行中显示红色「结束」按钮。
点「结束」会**在下一个进度回调（每页一次）停下来**，已完成的页面和 OCR 缓存都保留，
重新开始会接着跑，不会从头再来。

执行条右侧有「暂停」按钮：点一下在当前页处理完后停住（模型留在显存里，已完成的页面全保留），
按钮变成「▶ 继续」，侧边栏圆点转金色显示「已暂停 · 停在第 n/N 页」。暂停期间照样能点「结束」。
暂停状态在重启程序后不会保留（会被标记成已中断），需要的话重新开始即可，缓存仍然复用。

「设置」页最下方有「退出程序」按钮：停掉本机服务（正在跑的任务会先被取消），
浏览器显示「已退出」后可以关掉标签页。下次双击 `start.bat` 重新启动。

## 目录

| 路径 | 内容 |
| --- | --- |
| `app\` | 上游源码 + `.venv`（Python 3.12.15，torch 2.8.0+cpu），2.45 GB |
| `data\` | `MLS_HOME`：settings、模型、OCR 缓存、transcript，4.92 GB |
| `_dl\ollama\ollama.exe` | Ollama 0.40.14（免安装，只留 vulkan 后端），0.12 GB |
| `ollama-models\` | Ollama 模型库（`huihui_ai/qwen3.5-abliterated:9b` 6.6 GB + `hy-mt2-1.8b:q8` 1.9 GB），7.92 GB |
| `_cache\` | `pythons\`（venv 的 `home` 指向这里，**不能删**）+ uv 缓存，0.10 GB |
| `_home\` | `USERPROFILE` 重定向后的壳目录 + tmp，0.02 GB |
| `output\` | 成品输出 |
| `_test\` | 只留 `pony_dev`、`pony_builtin` 两份参考输出，0.03 GB |
| `_notes\` | 9 个常用脚本（`run_dev_job.py`、`run_builtin_job.py`、`rerender_pages.py`、`contact_sheet.py`、`fetch_hf_resume.py`、`probe_cores.py`、`probe_cpu_usage.py`、`probe_unconf.py`、`check_i18n.py`） |

整个目录 **15.2 GB**（清理前 49.36 GB）。模型部分：paddleocr ~93 MB、manga-ocr ~850 MB、
big-lama 196 MB、hy-mt2 3.81 GB（磁盘上有，当前配置改由 Ollama 服务，不再是必需项）、
ollama qwen3.5-9b 6.6 GB + hy-mt2-1.8b:q8 1.9 GB。

## 为什么必须在 ASCII 路径（重要）

Paddle 的 C++ 推理和 `torch.jit.load` 都**打不开含中文的路径**，会报
`json.exception.parse_error.101` / `open file failed because of errno 2`。
所以安装目录名 `Manga Localizer----Ben-Ding-Fan-Yi` 是纯 ASCII 的，**不要再改成中文名**。

之前目录在 `E:\工作区\漫画\Manga Localizer--本地自动翻译` 时，靠 `E:\mls` junction 绕开这个问题；
搬到 D 盘并改成 ASCII 名之后 junction 已删除，`start.bat` 全部用 `%~dp0` 相对路径。

配套的两处上游改动（不依赖 junction，但仍然是必需的历史修复）：

- `app\src\manga_localizer\config.py` 的 `AppPaths.from_env()` 去掉了 `.resolve()`。
  留着的话，即使传入 ASCII 路径也会被解析回真实中文路径，前功尽弃。
- `.venv\pyvenv.cfg` 的 `home` 和 `site-packages\_editable_impl_manga_localizer_studio.pth`
  指向 `_cache\pythons\...` 和 `app\src` 的绝对路径 —— **搬动目录后必须手动改这两处**，
  否则 python.exe 报 `No Python at '...'`、CLI 报 `No module named 'manga_localizer'`。

## 与原版 `start-windows.bat` 的区别

原版会调 `scripts\bootstrap.ps1`，在 AMD 机器上会装 CUDA 版 torch 且强依赖 GitHub 直连。
`start.bat` 跳过引导，直接用已装好的 CPU 环境，并把 `USERPROFILE`/`TEMP` 挪到 `_home`
（paddle 会往 `~/.cache/paddle` 写，C 盘只剩 24 GB）。

挪 `USERPROFILE` 的副作用：网页里点「浏览」选文件夹时，系统对话框以
`%USERPROFILE%\Desktop` 作默认位置，目录不存在就弹「位置不可用」。
`start.bat` 启动时自动补齐 `_home` 下的 Desktop/Documents/Downloads/Pictures/
Music/Videos/Favorites/Links/Searches/Contacts 十个壳目录。

**改 `start.bat` 必须保持 CRLF 行尾**：写成 LF 后 cmd 会把整行拆碎，
报 `'Localizer' is not recognized as an internal or external command`。

## 已知限制

- **OCR 是纯 CPU**（RX 9070 XT 只能给 Ollama 用 Vulkan，Paddle 没有 ROCm/DirectML 后端）。
  整本 71 页质量模式实测 **256 s**（约 3.6 s/页，含 OCR + 翻译 + LaMa 补画 + 排字）。
- 翻译已全部上显卡：9B 主译 + Hy-MT2 候选都由 Ollama/Vulkan 跑，不再走 CPU torch。
- `--ocr-backend hybrid` 会调 Ollama 做全页视觉语义。整页 VLM 会把网点纸、面板边框读成
  「ただだ」「効果か:」这类假文字，或把已有台词重读歪一两个字。本地已加三道过滤
  （`unconfirmed` 标记、`_rereads_known_text`、`repeat_penalty` 重试），实测 26 页能跑通；
  但它比 `builtin` 慢好几倍，日常建议用 `builtin`。
- 专有名词（书名、人名）单次翻译容易误译；质量模式已启用 Hy-MT2 候选 + 9B 审校，仍不保证。
- **白描边小字（`outlined` 且字号 < 40）擦除后仍可能留原字残影**：掩膜只覆盖描边内的笔画，
  LaMa 补出来还是原底色。已把掩膜边缘连通域的判定放宽，大部分能擦净；仍不净时改用
  `quality_profile=fast`（OpenCV 擦除）或手动框。

## 显卡加速（Hy-MT2 走 Ollama）

质量模式原先要 CPU torch 跑 Hy-MT2 1.8B 候选，8 页里占 172 s / 642 s（91% 的翻译时间）。
现在把**同一份权重**转成 GGUF 交给 Ollama，跑在 Vulkan 上：

```
_dl\hy-mt2\Hy-MT2-1.8B-Q8_0.gguf   1.82 GB（tencent/Hy-MT2-1.8B-GGUF，sha256 5c3fe0b1...）
ollama create hy-mt2-1.8b:q8 -f _dl\hy-mt2\Modelfile
```

`start.bat` 里 `set "MLS_HY_MT2_OLLAMA=hy-mt2-1.8b:q8"` 打开这个开关；删掉该行即退回 CPU torch。

实测（同一 8 页，质量模式）：**642 s → 83 s**（Hy-MT2 单次 9~74 s → 0.07 s）。
整本 71 页质量模式：**22.5 min → 4.3 min**。
译文基本一致，只有个别语气词差异（`とて、` CPU「于是，」/ GPU「真是的，」）。

## 本地修改记录（相对上游 v0.5.2）

`app\src\manga_localizer\config.py`（11 行）：

- `AppPaths.from_env()` 去掉 `.resolve()`（见上）。

`app\src\manga_localizer\renderer.py`（约 80 行）：

1. `_analyze_style` 竖排判定增加 `stacked_lines` 例外 —— 多个横向 erase_box 说明是横排多行文字，
   不再因为 `height >= width*0.62` 误判成竖排（木牌/标题框糊成一团的根因）。
2. 字色极性改按局部灰度中位数比较（`darker_ratio` vs `brighter_ratio`），
   不再用 `mean<145 and white_ratio<0.35` 这种会把青底黑字判成白字的启发式；
   白描边字（`halo_ratio >= 0.38`）固定为黑字白边。
3. `stacked_lines` 时字号上限收敛到 `height / 行数 * 0.62`，避免大字溢出牌子边框。
4. `_text_mask` 的种子阈值改中位数锚定（`level ± 45`），深底/浅底都能选出笔画。
5. `_erase()` 的极性也改按 `_interior_component_mask` 面积比较（原先按原始像素数，
   背景占比大时会选错极性，导致字没擦掉）。
6. `_text_mask` 放宽触边连通域：检测框切过字形时笔画必然触边，只在面积 > 25% 时才当画作丢弃
   （原先一律丢弃 → 整块文字残留）；`max_area` 从 0.16 提到 0.45。

`app\src\manga_localizer\translator.py`（约 20 行）：

- `_normalize_translation()` 剥掉模型自己加回的 `成稿：/定稿：/译文：/翻译：` 前缀
  （审校 prompt 里要求「输出…成稿」，小模型会把标签抄进译文）。
- `_source_is_question()` 不再把「句末 か + 任何标点」一律当问句。新增 `RHETORICAL_KA_RE`
  排除方言强调（`でっか`）、附和（`じゃないか`/`ではないか`）和自言自语（`てやろうか`）。
  原先 `ってでっか！！`、`身体は正直じゃないか．．．` 被判成问句，译文必须带「？」/吗/么/吧
  才放行，9 次重试全被拒 → `zh` 留空 → 整本抛 `Localization has N unresolved text units`。
- `_source_negation_contract()` 先把 `RHETORICAL_NAI_RE`（`じゃないか`/`ではないか`，后面不是问号）
  剥掉再判断否定。`身体は正直じゃないか` 是「身体很诚实嘛」的附和，不是否定句，
  原先强求中文出现「不/没」把「呵呵，身体倒是很诚实呢」整类译文拒掉。
- `LEXICAL_NAI_RE` 补 `無理(?:に|やり|矢理)`。`無理に/無理やり` 是「强行/硬要」不是「不可能」，
  原先被当成真否定，要求译文出现「不/没」，`ならば魔法で無理にでも…させてやろうか`
  的 8 次候选（「那就用魔法强行让她产生那种心情吧」等）全被拒 → `zh` 留空。
  真正的 `それは無理だ`（不可能）仍然保留否定要求。

`app\src\manga_localizer\renderer.py` 竖排补两条（见上）：

- 竖排分栏改为**均分**（`chunk = ceil(len/count)`）而不是按 `capacity` 贪心填满，
  否则最后一栏只剩「！！」这种孤立标点。
- `_fit_vertical` 的兜底从「固定 20px」改成继续往下缩到能塞进 `width`（下限 8px），
  小圆气泡原来会有一栏被 bbox 裁掉，读起来像丢字。

`app\src\manga_localizer\api.py` 加一个 `revalidate_web_assets` 中间件：给 `/`、`*.js`、`*.css`、`*.html`
发 `Cache-Control: no-cache, must-revalidate`。UI 没有构建步骤也没有内容哈希，
浏览器会启发式缓存 `i18n.js`，升级后仍渲染出 `settings.quit.title` 这类原始 key。

`app\src\manga_localizer\ocr.py`（约 45 行）：设备回落 + hybrid 误报过滤。

- `PaddleMangaOCR._resolve_device()` 增加 `_paddle_gpu_available()` 判定：设置里选「GPU 0」但
  Paddle 没有 CUDA 时回落到 `cpu`。原先直接把 `"gpu:0"` 透传给 Paddle，它只警告一句
  「device not available, switching to CPU」然后死在 oneDNN 的 PIR 转换里
  （`ConvertPirAttribute2RuntimeAttribute`），任务 0% 就失败。
- `HybridMangaOCR.analyze()` 把「几何恢复后仍未被精确检测框确认」的 claim 打上
  `unconfirmed=True` 而不是丢弃。整页 VLM 常把网点纸读成「ただだ」、把面板边框读成
  「効果か:」，这些 claim 没有检测器几何，既擦不掉也放不下译文；原先每一条都会让整本书失败。
  标记后仍写进 transcript 和 work 目录留档，`pipeline.unsafe_semantic_missing()` 跳过它们。
- 新增 `_rereads_known_text()`：召回阶段会重读检测器已经拥有的台词并读歪一两个字
  （`…ちゃんと話を…` → `…ちゃんとなんて話を…`、`…平気よ` → `平気よの`），精确包含判据抓不到，
  于是整页报「未覆盖文字」。改用 bigram 覆盖率 ≥ 0.6 或 `difflib` 相似度 ≥ 0.5 判为「重读」，
  长度 < 3 的短句豁免，避免把「がー！」这类真音效吞掉。
- `OllamaVisionOCR._chat()` 重试时加 `repeat_penalty=1.3` / `repeat_last_n=256`。第 5 页曾让 9B
  陷入退化循环（反复输出 `オ` 且 bbox 坐标无限增长），budget 8192→12288→16384 全被撑爆；
  加了重复惩罚后 965 token 就正常收尾。

`app\src\manga_localizer\inpainting.py`（约 8 行）：`LaMaInpainter._load()` 同理 ——
请求 `gpu`/`cuda` 但 `torch.cuda.is_available()` 为假时回落到 `cpu`，否则抛
`Torch not compiled with CUDA enabled`（本机是 AMD 卡 + CPU 版 torch）。

`app\src\manga_localizer\translator.py` 另加（约 6 行）：`HyMTTranslator._ensure_loaded()` 在
`pytorch_device()` 之后做同样的 CUDA 可用性检查。**没有改 `pytorch_device()` 本身**，
因为 `tests/test_translator.py` 断言 `pytorch_device("gpu:0") == "cuda:0"`。

`app\src\manga_localizer\pipeline.py`（约 12 行）：

- ollama 分支的 Hy-MT2 候选支持 `MLS_HY_MT2_OLLAMA=<ollama tag>` 环境变量，
  有值时用 `OllamaTranslator` 代替 `HyMTTranslator`（同一权重，跑在显卡上）。
- `unsafe_semantic_missing()` 跳过 `unconfirmed` 标记的 claim（见上）。

`app\src\manga_localizer\web\*`（约 30 行）：页面预览下方加了**页码滑动条**。

- `index.html` 在 `.pager` 之后加 `<div class="page-slider"><input id="pageSlider" type="range">`，
  右侧显示 `n / total`。
- `app.css` 把 `.preview-card` 的 `grid-template-rows` 从三行改成四行（末行 `40px` 给滑动条），
  轨道用 `--fill` / `--tick` 两个 CSS 变量画进度色与刻度点，滑块 20px 圆点。
- `app.js` 加 `fillPageSlider()` / `syncPageSlider()`：`renderPreview()` 末尾同步，
  `input` 事件只更新页码和图片、不重排整页；只有一页时滑动条禁用。
- `resumeJob()` 在刷新页面且没有运行中任务时，自动载入**最近一个已完成**的 job，
  这样滑动条和预览不是空的。它走的是 `openJob()`，所以「源图片文件夹」「输出目录」
  也会一起回填（原先直接 `renderJob()`，刷新后两个路径框是空的）。

`app\src\manga_localizer\web\*`（约 25 行）：任务记录页的刷新反馈与路径回填。

- 「刷新」原来只重绘列表、没有任何提示，看着像没反应。现在 `refreshJobs()` 返回条数，
  按钮在请求期间禁用，完成后弹 `已刷新，共 N 条记录`。
- 卡片时间戳原来是原始 ISO（`2026-11-04T08:43:10.332767+00:00`），新增 `formatJobTime()`
  转成本地 `YYYY-MM-DD HH:MM`。
- 点记录原来只 `renderJob()` + 切视图，**不回填表单**。新增 `openJob(jobId)`：把
  `job.request.source` / `job.request.output` / `job.request.reviewed_transcript`
  写回 `#sourcePath` / `#outputPath` / `#reviewedTranscript`，页码归 1，再渲染。
- `renderJob(job,{quiet:true})` 用于从历史打开：失败的任务不再弹红色 toast
  （预览区会显示「任务失败，暂无译文页面」），只有新任务真的失败时才提示。
- `.utility-head .secondary-button{width:auto}` 覆盖全局 `width:100%`，刷新按钮回到标题右侧
  （原来横贯整行）。
- 三语言补 `history.refreshed`。
- 新增测试 `test_history_cards_restore_the_job_paths`。
- 刷新页面时 `resumeJob()` 现在也走 `openJob()`，所以两个路径框不再被清空；
  新增测试 `test_refresh_restores_the_last_job_paths`。

`app\src\manga_localizer\jobs.py` + `api.py` + `web\*`（约 90 行）：侧边栏任务状态栏与「结束」按钮。

- `jobs.py` 增加 `JobCancelled` 与合作式取消：`JobService._stop_flags` 存每任务的 `threading.Event`，
  `cancel(job_id)` 置位并把状态写成 `cancelled`；`_run()` 里的 `progress()` 每次回调先查标志，
  已置位就抛 `JobCancelled`，`except JobCancelled` 分支保住 `cancelled` 状态不被 `failed`/`complete` 覆盖。
  **取消点是进度回调，所以最快在下一次回调（每页一次）生效**；已缓存的 OCR 结果保留，重跑接着走。
- `api.py` 增加 `POST /api/jobs/{job_id}/cancel`（未知 id 返 404）。
- Web UI：侧边栏左下角新增 `#jobStatus` 卡片（状态圆点 + 标题 + `百分比 · 当前/总页数` + 「结束」按钮），
  颜色随 `data-state` 变化（idle 灰 / running 橙色脉冲 / complete 绿 / failed 红 / cancelled 灰）。
  页面刷新时 `resumeJob()` 会找回正在运行的任务继续轮询。
- 三语言补 `job.idle / job.cancelled / job.cancelledDetail / job.cancel / job.cancelConfirm / status.cancelled`。
- 新增测试 `test_cancelled_job_stops_and_is_not_marked_complete`、`test_cancel_unknown_job_returns_404`。

`cli.py` + `api.py` + `web\*`（约 45 行）：设置页「退出程序」按钮。

- `cli.py` 的 `ui()` 改用 `uvicorn.Server(uvicorn.Config(...))` 并挂到 `application.state.server`，
  这样前端能拿到服务句柄（`uvicorn.run()` 会把它藏起来，Windows 上也没有可用的 SIGINT）。
- `api.py` 增加 `POST /api/shutdown`：先把 `queued`/`running` 的任务全部 `cancel()`，
  再延迟 0.4 s 把 `server.should_exit` 置真（先让 HTTP 响应发出去）。
  **必须先取消任务**——`ThreadPoolExecutor` 的线程不是 daemon，任务还在跑时 uvicorn 停了进程也不会退。
  没有 server 句柄时（测试/嵌入场景）返 409。
- Web UI：设置页底部 `#quitApp` 按钮 + 确认框，成功后把页面替换成「已退出」提示。
- 三语言补 `settings.quit.title / detail / button`、`settings.quitConfirm`、`settings.quit.done / doneDetail`。
- 新增测试 `test_shutdown_requires_a_server_handle`、`test_shutdown_stops_the_server_and_cancels_running_jobs`。

`app\src\manga_localizer\jobs.py` + `api.py` + `web\*`（约 110 行）：安全暂停。

- 取消用的 `threading.Event` 换成 `_JobGate`（`threading.Condition` + `_paused`/`_cancelled`）。
  `progress()` 第一行 `gate.wait_while_paused()`：**暂停就停在进度回调上**，也就是页与页之间
  （OCR/渲染）和 chunk 与 chunk 之间（翻译），不会留下半张擦过字没排字的图。模型仍常驻内存，
  已写进 work 目录的页面全部复用，`resume()` 从同一页接着跑，不重头开始。
- 暂停中取消也走得通：`gate.cancel()` 会 `notify_all()` 唤醒阻塞的 worker 并抛 `JobCancelled`。
- `LIVE_STATUSES = {"queued","running","paused"}` 统一给 `cancel()` 和 `/api/shutdown` 用；
  暂停的任务也必须在退出时取消，否则非 daemon 线程会卡住进程。
- 新增 `_adopt_orphans()`：进程启动时把上次遗留的 `queued`/`running`/`paused` 全部标成
  `cancelled`（"服务重启，任务已中断"）。没有 gate 的暂停任务谁也 resume 不了，留着只会骗人。
- `api.py` 增加 `POST /api/jobs/{job_id}/pause` 与 `/resume`（未知 id 返 404）。
- Web UI：`#pauseButton` 从永久 disabled 变成「Ⅱ 暂停 / ▶ 继续」切换，`syncPauseButton()`
  在 `renderJob()` 和语言切换时同步；暂停后按钮变实心主色，侧边栏圆点转金色（`data-state="paused"`），
  窄屏下按钮不再被 `display:none` 隐藏。`ACTIVE_STATUSES` 加 `paused`，轮询不会停。
- 三语言补 `job.pauseDetail / job.resume / job.paused / job.pausedDetail / job.pausedToast /
  job.resumed / status.paused`，删掉 `job.pauseUnsupported`。
- 新增测试 `test_paused_job_holds_its_page_until_resumed`、`test_paused_job_can_still_be_cancelled`、
  `test_pause_and_resume_unknown_job_return_404`、`test_leftover_paused_job_is_cancelled_on_startup`、
  `test_pause_button_toggles_between_pause_and_resume`。

`app\src\manga_localizer\web\app.js` + `api.py`（约 15 行）：刷新后路径被清空 + 源目录已删时预览 500。

- 页面刷新时 `resumeJob()` 会找回任务，但原来直接 `renderJob()`，**不碰表单**，所以「源图片文件夹」
  「输出目录」两个框是空的。现在统一走 `openJob(id)`，路径、`reviewed_transcript`、页码一起回填。
- `api.py` 的 `preview` 把 `list_images(source)` 的 `ValueError` 转成 404（源目录被删/改名后
  控制台原来会刷一串 500）。新增测试 `test_preview_of_a_deleted_source_folder_is_a_404`。

`app\src\manga_localizer\jobs.py` + `web\*`（约 40 行）：侧边栏任务卡片加计时。

- `jobs.py` 的 `JobStore.save()` 是所有状态变更的必经之路，时钟簿记就放这里一处：
  `running` 首次出现记 `started_at`，`paused` 记 `paused_at`（`resume`/取消/结束时会折叠进
  `paused_seconds`），终态（`complete`/`failed`/`cancelled`）记 `finished_at`。这样**暂停时段不计入
  已用时间**，恢复后计时从暂停点继续。
- 侧边栏卡片在状态行下新增一行 `#jobClock`：`开始 22:14 / 已用 0:08 / 预计还需 1:37`，
  预计 = 已用 × (100-progress) ÷ progress（进度 ≥3% 才显示，避免开头抖动）；任务结束显示 `共用时`。
  轮询本来就每秒一次，计时随 `renderJob()` 自然跳动，不用另开定时器。
- 三语言补 `job.clock.start / elapsed / eta / total`。
- 新增测试 `test_job_records_its_clock_and_ignores_the_pause_gap`、`test_sidebar_shows_the_job_clock`。

`model_manager.py` + `api.py` + `cli.py` + `web\app.js`（约 130 行）：模型页列的是**实际在跑**的模型。

- 原来的 4 行只是「磁盘上的权重」，真正的翻译模型 `huihui_ai/qwen3.5-abliterated:9b` 根本没显示；
  质量模式的第二遍也不是磁盘上的 `data\models\hy-mt2`，而是 `MLS_HY_MT2_OLLAMA=hy-mt2-1.8b:q8`
  这个 Ollama tag。现在 `/api/models` 会连 `{ollama_base_url}/api/tags`，把这两个 tag 作为
  `kind="ollama"` 的行追加在后面，带真实大小（`~6.1 GB` / `~1.8 GB`）和 `已拉取/未拉取/服务未连接`。
- 连不上 Ollama 时这些行 `ready=false`、状态显示「服务未连接」，不会假装就绪。
- Ollama 行不参与「所需模型已就绪 / 准备缺失模型」的判定（`isLocalModel()`），因为 tag 由任务按需拉取，
  不是 `data\models` 下的文件，也不能走 `/api/models/bootstrap` 下载。
- 改由 Ollama 服务后，磁盘上的 `Hy-MT2 1.8B` 权重不再是必需项（`required=false`，显示「当前后端不需要」）。
- 顺带修了两个错数字：paddleocr `~180 MB` → `~93 MB`（实际 92.5 MB）、manga-ocr `~450 MB` → `~850 MB`（实际 847 MB）。
- Ollama 行的角色/许可/状态文案走后端 `role_key`/`status_code` + i18n（`models.role.translation|review|ocr`、
  `models.license.upstream`、`models.pulled|notPulled|offline`），三语言齐。
- `cli.py doctor` 也接上了 runtime，`dsh doctor` 输出同样能看到 Ollama 实际模型。
- 新增测试 `test_models_page_lists_the_ollama_tags_that_actually_run`、
  `test_models_page_says_so_when_ollama_is_not_answering`；`test_health_models_and_job_flow` 改为只数
  `kind=="local"` 的行并 monkeypatch 掉 `ollama_tags`（否则测试会去连本机 11434）。

`web\app.js` + `web\index.html` + `web\app.css` + `web\i18n.js`（约 60 行）：全站功能巡检后的 7 项修补。

- **Ollama 掉线不再挡住「开始本地化」**：`updateStartState()` 的 `missing` 判定补了 `isLocalModel()`，
  与 `renderModels()`/`bootstrapModels()` 一致。此前 Ollama 服务没起来时按钮会被误禁并提示
  「请先准备缺失模型」，而实际上 tag 是由任务自己按需拉的。
- **任务失败后任务记录会刷新**：`pollJob()` 原来只有 `complete`/`cancelled` 分支，`failed` 时列表不更新。
- **暂停状态在记录页有专属配色**：`refreshJobs()` 的 chip class 补 `paused`，配 `.status-chip.paused`。
- **暂停时流程条保持高亮**：`renderPipeline()` 的 `active` 判定加 `paused`，`#pipelineEstimate` 显示
  「已暂停」而不是停在最后那个百分比。
- **模型管理页有刷新按钮**：新增 `#refreshModelsLarge`（↻），与「准备全部模型」并排；
  配 `.utility-head .head-actions` 让两个按钮靠右不被 `width:100%` 拉满。
- **改设置会提示「设置已保存」**：`saveSettings()` 成功后 `toast(t("settings.saved"))`。原来静默保存，
  用户不知道有没有生效。三语言补 `settings.saved`。
- **「测试推理连接」防重复点击**：原来按钮不 disabled，在线后端 30 s 超时期间可以连点发出多次请求。
  现在 `state.inferenceToken` 自增做令牌，请求中禁用按钮，过期的响应直接丢弃。
- 新增测试 `test_ui_polish_keeps_ollama_tags_and_paused_state_honest`（12 条断言覆盖上面 7 项）。

`api.py`（2 行）：自动推导的输出目录名改为「源文件夹名 + `--已汉化`」，且可选地落在用户选定的输出根目录下。

- `POST /api/paths/derive-output` 现在收两个可选参数：`initial`（源目录）与 `output`（输出根目录）。
  - 只传 `initial`：返回 `{源目录名}--已汉化`，跟源文件夹放在同一个父目录下（用户选完源目录、输出框还空着时的旧行为）。
  - `initial` + `output` 都传：返回 `{输出根目录}\{源目录名}--已汉化` —— 翻译过程中的所有文件（`.manga-localizer-work` 中间产物、成稿）都会落在该子文件夹里。
- 前端 `pickFolder` 在选完「输出目录」且源目录已填时也会调用 `deriveOutput()`，把输出框自动改成 `{输出根}\{源名}--已汉化`。
- `tests\test_api.py::test_derives_non_destructive_output_path` 的断言同步改成 `chapter-01--已汉化`，
  新增 `test_derives_output_under_the_picked_output_root`；`test_web_contract.py` 新增
  `test_derive_output_lands_under_the_picked_output_root`。

`app\src\manga_localizer\translator.py`（约 10 行）：**「已可接受的译文是终稿」守卫**。

- 实测发现质量模式末轮（`judge_candidates` / `retranslate_risk_units`）会把编辑初译的好译文
  覆盖成更差的版本：`translation_attempts` 里前三次都是同一个好译文（editor/review 稳定输出），
  第四次来自审校/重译轮（如把后一格的「2015 年 10 月 14 日」串进 `まぁ、そういったくなったのですから、`
  的译文、丢「那就…开始吧」、`サイト` 错译成「网站」），且因为新候选能过硬门槛就被直接采纳。
- 修复：这两处覆盖点都加 `and not self._candidate_is_acceptable(unit, unit.zh, glossary)` ——
  后期 pass 只能救援被硬门槛拒绝的行（假名泄漏、否定/问句不匹配、长度越界等），
  **不得覆盖已可接受的译文**。`review_pages` 保持原样（它有上下文拆句修复职责，
  测试 `test_context_review_repairs_split_sentence_without_changing_unit_mapping` 证明其价值）。
- 新增回归测试：`test_retranslation_never_overwrites_an_accepted_translation`（坏候选只追加 attempt
  不覆盖 zh）、`test_retranslation_still_rescues_a_rejected_line`（假名泄漏行仍会被救）、
  `test_judge_never_overwrites_an_accepted_translation`。

### 英文源漫画（非日语）支持

原来的流水线只有日语一条路：OCR 侧用 `manga-ocr`（只会日语，看英文气泡会**编出假名**），
翻译 prompt 写死「日语漫画」，效果器表只收假名声效。拿一本英文同人试跑，
整本被读成日语幻觉、译文全错。改动如下。

`app\src\manga_localizer\ocr.py`：

- `_regions()` 优先用 recognition 分支（`if len(payload.get("rec_boxes") or [])`），
  面积门槛从「一律 650 px²」放宽为 `_area(box) < 650 and not (score >= 0.8 and len(text) >= 2)`
  —— 气泡里的小字（`um,` / `TH-`）只有约 400 px²，被硬门槛丢掉后，气泡首尾行会保持英文
  而中间被翻译。`_area / len(compact) > 2_000` 才判为「大片区域里零星一个字」并清空。
- 新增 `is_latin_page(regions)`：整页拉丁字符数 > 中日文字符数×2 即为英文页。
  `analyze()` 在 `_groups()` 之前把英文页里**没有文本的框**全部剔除 —— 英文页 detector
  的无文本框是画面（鬃毛、衣服、面板边框），既把 unit bbox 撑到画面大小（排版算出巨大字号、
  文本溢出气泡，quality 模式还会把这块画面用 LaMa 擦掉），又会把两个独立气泡桥接成一个
  unit（旁白和台词被当成一句翻译）。参见 `p005u12`：`bbox=[275,919,499,1103]` 里
  只有 `[275,919,357,940]` 是真文字，`[339,966,499,1103]` 是画。
- `pipeline.py` 的 `OCR_CACHE_VERSION` 升到 `8`：v7 的缓存里存的是 MangaOCR 对英文页的
  日语幻觉，不失效会一直复用坏结果。

`app\src\manga_localizer\translator.py`：

- 新增 `latin_only_source(text)` / `source_language_label(pages)`（kana+汉字为 0 且只有拉丁
  → `英文`，否则 `日文`）。`PromptTranslator._language()` 在 `translate_pages` 里定语言，
  `_script_rule()` 返回「日文假名」或「英文原文」；11 处写死的「日语漫画」「日文假名」
  「日文效果字」改成 `{self.source_language}` / `{self._script_rule()}`。
  `LocalQualityTranslator` 会把语言**逐个推给** editor / candidate / auditor 三个角色
  （auditor 不走 `translate_pages`，只推一次不够）。
- `_valid_translation` 新增 `untranslated_latin` 判定：英文源下 `SOURCE_LATIN_RE` 变宽松，
  模型整句回吐原文也能过关，所以译文必须换脚本，最多留 2 个拉丁词（专名），
  `Display 展现 oPassion 6 年 Alcor` 这种半翻译会被拒。
- `_preserves_audit_information` 的长度下限对英文源额外按 `round(len(source) * 0.15)` 兜底
  —— 英文没有汉字，原来的汉字密度公式会退化成「3 个字就够」，一句话摘要就能过。
- `_translate_sfx` / `_valid_sfx_translation` 的正则加 `A-Za-z`，否则英文拟声词清洗后为空、
  长度上限算成 4。

`app\src\manga_localizer\sfx.py`：新增 `_LATIN_SFX` 表（25 条）、`_latin_sfx()` 与
`_latin_sfx_phrase()`，在 `lookup_sfx()` 最前面查。覆盖 `grr/argh/augh/ugh/hmph/gasp/sigh/
poof/bam/bang/slam/thud/crash/clang/knock/creak/rustle/splash/gulp/munch/sniff/wheeze/zzz`，
并把末字符重复的形式归并（`GRRRRR` → `grr`）。否则 `Grrrrr` 会被音译成「格勒勒」。
英文页常把一个效果字写两遍（`sniff snif`、`poof poof`），所以 `_latin_sfx_phrase()` 会把
2–4 个词逐个查表、允许 OCR 掉一个字母（`nif` ⊂ `sniff`，`_effect_word()`），全部命中才算音效，
并把重复的中文叠起来（`吸溜吸溜`、`噗噗`）。`ocr.py` 的 `likely_sfx_text()` 同步放宽为
`[A-Za-z]{2,14}(?:\s+[A-Za-z]{2,14}){0,3}[!?！？.…]*`。

`app\src\manga_localizer\renderer.py`：`_fit_horizontal` 的字号下限从 20 px 降到 8 px
（`range(start, 7, -2)` + 兜底 `_font(font_path, 8)`）。一行英文译成中文后字数暴涨，
停在 20 px 会溢出气泡。

新增测试：`test_single_glyph_spread_over_artwork_loses_its_text`、
`test_latin_page_drops_text_less_artwork_boxes`、`test_latin_sound_words_are_effects_but_latin_dialogue_is_not`、
`test_repeated_latin_sound_words_are_one_effect`、`test_link_blocks_are_preserved_instead_of_translated`、
`test_latin_sources_are_not_measured_with_japanese_prose_gates`、
`test_unresolved_units_are_preserved_instead_of_failing_the_book`、
`test_english_book_is_labelled_and_cannot_echo_the_source`、
`test_prompts_follow_the_source_language`、`test_half_translated_english_line_is_rejected`。

真机验证（5 页英文同人，builtin OCR + 9B Ollama + quality）：修复前封面上仅有的
`Display ofPassion by Alcor` 被译成半英文，正文整段 `FOR TODAY IS THE DAY...` 被读成假名；
修复后逐句可读（`今日，我将向那几近遗忘的文化传统，向我那失落的故土，致以应有的敬意。`），
`Grrrrr`/`AUGH!`/`poof` 按音效保留原文，两个气泡不再被合并。
双写的拟声词 `sniff nif` 现在也被认成音效（保留原文），不再被译成「嗅嗅，尼夫」。

英文源的已知限制：**没有气泡、直接画在背景上的小字可能漏检**。实测
`00000005.webp` 左下角那行 `Daddy, what "in heat" means?`（约 53×10 px 两行）在默认
（原分辨率）下 detector 与 recognition 都读不到，输出里就保持英文。把 PaddleOCR 的
`text_det_limit_type="min"` + `text_det_limit_side_len=1280`（短边放大 1.6×）能把它读出来，
但整页 recognition 从 ~26 s 涨到 ~34 s（+32%），并且会多出一批噪声框
（`iOTTH` s=0.95、`oop fig` s=0.53、`UOISSD` s=0.58 等），这些框面积够大、会被当成台词
翻译后画到画面上。所以**没有默认开启**；要提高小字召回，就得接受这个代价。

### 「Localization has N unresolved text units」：从整本失败改成保留原文

英文页有一种很致命的失败方式：任务跑到 70% 抛
`Localization has 14 unresolved text units (p007u02, p022u07, …)`，然后**整本不出成稿**。
这是 `pipeline.py` 收尾的硬门槛（`completion_summary()` 的 `unresolved_units != 0` 就 `raise`）：
只要有**一个** unit 的 `zh` 为空、或过不了 `_valid_translation`，
前面 OCR + 翻译的二十多分钟全部白跑。用那本 74 页英文书复现（`_notes\probe_unres.py`
直接读已落盘的 OCR 缓存跑同一套 quality 栈），先修掉三个「拿日语标准去量英文」的门槛：

- **字形下限**：`_preserves_audit_information` 对 `FINALLY` 算 `max(3, round(7×0.15)=1)=3`，
  正确译文「终于」只有 2 字被拒；`Shlap`（SLAP 的误读）更早就被
  `len(source) >= 5 and len(proposed) < 2` 拒掉（正确译文「啪」只有 1 字）。
  现在英文源的下限是 `max(1, round(len(source) * 0.15))`，那个两字形门槛只对非拉丁源生效。
- **不许乱加感叹号**：源是 `NGGHH...` / `NNNN-NN` 这类呼喊时，译文「啊……哈！」「啊！再久点……」
  被 `source_is_emphatic` 判成无端激动而拒。现在**长度 ≤ 8 的拉丁单词**豁免这条（长句仍受管）。
- **120 字上限**：`_valid_translation` 的 `min(120, len(ja) * 4)` 是按日文散文定的；英文页一屏小字
  会被 OCR 合并成一个 557 字的 unit，忠实译文 200+ 字，全部撞上限。现在英文源的 ceiling 取
  `max(ceiling, round(len(ja) * 0.75))`。
- **版权页的链接块**：`p073u03` 是后记里的一串 `https://…`（Patreon / Discord / Twitter），
  模型要么原样回吐（不换脚本 → 拒），要么写成摘要（原文含 `/` 也会被 `context_leak` 拒），
  于是这个 unit 永远无法通过、整本失败。`ocr.py` 新增 `link_block_text()`（≥3 个链接，
  或链接占去一半以上字符），判定放在 `classify_non_story_text()` 里 —— 那是 `pipeline.py`
  对所有后端、**包括复用缓存页**统一执行的「非正文」契约，所以标记一定生效，
  命中就把该 unit 标成 `skip_reason="preserve"`、`special="link_block"`：
  渲染阶段跳过它，**原图的链接保持原样**。
  （第一版把判定插进了 `OllamaVisionOCR` 的收尾循环，而任务走的是 `builtin` 后端，
  那个分支根本不执行 —— 规则必须放在两个后端共用的契约里。）

然后再给收尾加**兜底**：`preserve_unresolved_units()`（`pipeline.py`）在翻译阶梯八次尝试
全部用尽后，把仍然过不了硬门槛的 unit 标成 `skip_reason="preserve"`、
`special="unresolved_fallback"` —— **原图那几行英文保持原样**，任务继续渲染并出成稿，
而不是整本失败；被保留的 id 写进 `translation_manifest.json`，任务进度里也会写一行
「N 处未能翻译，已保留原文：…」：

```json
"quality": {"total_units": 1018, "translated_units": 1015, "explicitly_skipped_units": 3,
            "unresolved_units": 0, "fallback_preserved_units": 3,
            "fallback_preserved_ids": ["p052u27", "p069u08"]}
```

之前那次是**随机失败**：同一本 74 页，第一轮挂 `p023u04/p052u27/p073u03`，第二轮挂
`p052u27/p052u28/p069u08` —— 合并出来的长英文块能不能过审计门槛取决于模型的发挥，
所以「修门槛」只能减少概率，必须配兜底才不会白跑一趟。

**最终真机结果**（job `1fe6857a9a52`，74 页，`complete 100% 74/74`，输出
`output\1416037-…--已汉化`）：74 个 webp 全部生成，`quality` =
`{"total_units": 1018, "translated_units": 1016, "explicitly_skipped_units": 2,
"preserved_sfx_units": 0, "unresolved_units": 0, "invalid_translation_units": 0,
"fallback_preserved_units": 1, "fallback_preserved_ids": ["p047u01"]}` ——
只有一处（`p047u01`，一串 OCR 读歪的呻吟 `GH TWI-- mh nh hh ah…`）保留了原文，
版权页链接块 `p073u03` 按设计保留，其余 1016 个 unit 全部译出。

日文源的 3 字形下限、120 字上限、感叹号规则**全部保留**（新增回归测试
`test_latin_sources_are_not_measured_with_japanese_prose_gates` 同时断两侧，
`test_unresolved_units_are_preserved_instead_of_failing_the_book` 断兜底）。

⚠️ 复现这个 bug 时注意：`.manga-localizer-work\translation-draft.json` 是**按输出目录共享**的，
被后续 run（哪怕是被取消的）覆盖过，只有 `0000000N.ocr.json` 缓存可靠。
另外**跑 `pytest` 会触发 `_adopt_orphans()`**（有些测试用真实 `MLS_HOME`），把正在跑的真实任务
标成「服务重启，任务已中断」——测试前先确认没有任务在跑。

`tests\test_inpainting.py::test_lama_wrapper_preserves_every_unmasked_pixel` 在本机失败，
原因是 `torch.jit.save` 打不开含中文的临时目录（pytest 的 tmp_path 落在中文目录下），
不是代码问题。其余 219 个测试全绿（共 220 个）：
`python -m pytest tests --deselect tests/test_inpainting.py::test_lama_wrapper_preserves_every_unmasked_pixel`

## 磁盘清理（2026-11-04，49.36 GB → 15.16 GB）

删掉的都是**确认无引用**的东西，删完跑过完整测试和一次真实两页任务。

- **Ollama 实验模型 4 个（13.3 GB）**：`deepseek-ocr:q8`、`hunyuanocr:q8`、`wevisone-vl-4b:q8`、
  `wevisone-vl-4b:q4km`。都是「OCR 上 GPU」失败实验的遗留，源码/测试/settings 零引用。
- **`_dl\` 下载缓存（14.1 GB）**：`deepseekocr`、`hunyuanocr`、`wevisone`、`WeVisOne-VL-src`、
  `onnx`、`hy-mt2`（源 GGUF，已 `ollama create` 进库）、`ollama-windows-amd64.zip`（已解压）。
  只留 `_dl\ollama\`——`start.bat` 依赖它拉起 `ollama serve`。
- **`_dl\ollama\lib\ollama\cuda_v12` + `cuda_v13`（2.03 GB）**：这台是 RX 9070 XT，走 `vulkan`。
  删完 `ollama run hy-mt2-1.8b:q8` 仍正常推理（13.9 s 首字）。
- **`_home\.cache\huggingface`（1.27 GB）**：旧 HF 缓存，`HF_HOME` 已指向 `data\models\huggingface`。
- **`data\models\huggingface\...\snapshots\82c77fb...`（423.5 MB）**：`refs\main` 指向的是
  `aa6573bd...`，这个 82c7 快照只被 `refs\refs\pr\4`（下载工具的临时引用）挂着，是孤儿。
  **注意权重只能用 `pytorch_model.bin`**——实测把它挪走、只留 `model.safetensors` 后，
  `MangaOcr()` 直接报 `OSError: ... does not appear to have a file named pytorch_model.bin
  or model.safetensors`（transformers 5.x 在这个仓库的 `config.json` 上不会自动退到 safetensors，
  它在 `.no_exist\` 里已经把 `model.safetensors` 标记为不存在）。所以删的是 safetensors 那份，
  `pytorch_model.bin` 保留。
- **`_home\.paddlex`（92.5 MB）**：与 `data\models\paddlex` 完全重复，`start.bat` 指向后者。
- **`data\models\onnx`（80.7 MB）**：源码零引用（`onnxruntime` 只是 paddle 的传递依赖）。
- **`_cache\uv`（3.02 GB）**：`archive-v0`、`sdists-v10` 和 15 个 `.tmp*` 残留。`_cache\pythons` 保留。
- **`_test\` 37 个目录（约 220 MB）**、**`_notes\` 309 个文件（23.2 MB）**、
  `data\probe1.txt`、`_speedtest.py`、`_pt\`。
- **`data\jobs\` 18 条 failed/cancelled 记录 + 1 条指向已删目录的**，只留 2 条 `complete`。

清完验证：`pytest` 199 passed / 1 deselected；`/api/models` 仍是 6 行；真实两页任务
（builtin + ollama + quality）跑通 `complete 2/2`。

## 已生成的成品

`output\雌豚の怪異、キスショットさま 1-3\`：71 页中文版（WebP，30.1 MB）。
