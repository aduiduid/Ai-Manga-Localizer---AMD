# Changelog

## 0.10.0 - 2026-11-08

- Read pages through llama.cpp as well: `ocr_backend="llama"` runs the same `HybridMangaOCR` geometry-then-semantics split, with `LlamaVisionOCR` in place of the Ollama vision adapter. Because only the transport is swapped, a page's boxes and text are directly comparable between the two backends; an A/B on the sample book matched 4/4 boxes and 4/4 readings on both pages, with identical skip and role verdicts.
- Convert pages to PNG for the vision request. The llama.cpp image loader rejects WebP, which is the only format this app stores, so every page would otherwise fail on the first image.
- Start the vision model on its own port (8082) and reuse the translation slot when it already holds the same tag. The main model is the multimodal one, so batch translation would otherwise load a second copy of the same 7 GB weights; the vision slot still gets its own port because `-np 1` serves one request at a time and OCR cannot queue behind translation.
- Bounds the confidence `score` inside the detection schema. llama.cpp compiles the JSON schema into a constrained decoding grammar, and an unbounded number let it answer `0.0` for every box on a page it had read perfectly -- which silently disabled the score gates downstream (`prefer_semantic_ocr` wants 0.72, `_local_crop_is_suspicious` rejects <= 0.05). The type is stated as `minimum: 0, maximum: 1`, exactly as `refine` and `find_missing` already stated theirs.
- Release the servers when either stage started one: a finished batch now stops the vision server even when translation stayed on the Ollama daemon.
- Add `llama` to both OCR pickers and the settings panel: a run whose OCR is on llama.cpp reuses the Ollama visual-model field for the vision tag, and the start button asks for it when either stage needs it.

## 0.9.0 - 2026-11-08

- Run the translation and review models on llama.cpp instead of Ollama. The GGUF blobs Ollama already downloaded are reused in place through hard links, so nothing is copied or converted: the manifest holds a standard GGUF v3 and Ollama's own runner is llama.cpp. Two servers are started (8080 for the multimodal main model, 8081 for the review model) because the bundled build has no router mode.
- Start the servers from the staged runtime directory. `ggml-vulkan.dll` ships in a `vulkan/` subdirectory that only Ollama knows how to load, so a `llama-server.exe` launched directly finds no GPU and silently falls back to CPU: the vulkan backend and its dependencies are hard-linked next to the executable first.
- Disable thinking for every request with `-rea off`. Qwen3.5 otherwise spends the whole budget on `reasoning_content`, returns an empty `content`, and looks like a truncation.
- Convert images to PNG before sending them. llama-server answers `400 Failed to load image or audio file` for WebP, which is what the manga pages are.
- Keep the model loaded for the whole batch and release it when the last holder exits. `hold_llama_servers()` wraps the batch consumer, `jobs._run` releases on every exit path, and `/api/shutdown` stops the servers because a detached child process outlives uvicorn and keeps the VRAM.
- Move to the next free port when 8080 or 8081 is taken, and say so in the run log instead of failing silently; a fully busy range fails with the range in the message.
- Add the `llama` inference backend end to end: settings, `/api/models` rows resolved from the Ollama manifests, `/api/inference/check` that validates the executable and both tags, and a setup panel with the model tags, context size, ports and the optional Ollama root.

## 0.8.0 - 2026-11-08

- Decide which side of the background is text from the detector box alone. The erase region is padded well past the text so stray furigana is reachable, but counting that padding let panel art and highlights outvote six glyphs: a pink sign with black lettering was treated as light-on-dark, the mask selected empty pixels, and the original Spanish stayed under the translation. The padded region is still erased, it just no longer votes.
- Stop shrinking translations below 12px. A Chinese glyph at 8px is a grey smear at manga resolution, so the typesetting region now grows first: into the balloon it sits on, and past the balloon when the bubble is genuinely too small. Overlapping a little artwork is a far smaller defect than text nobody can read. Across the four sample folders the smallest size went from 8px to 12px and the mean from 14.1px to 15.9px.
- Wrap Latin words whole and never start a line with closing punctuation. "来得，/Twilight/，" used to break inside "Twilight" and push a lone comma onto a third line; "Twilight，" now stays one unit, trailing marks hang into the margin, and a one-glyph line gives up a size instead of hanging.
- Tighten line pitch and the box inset so short balloons keep their type: 1.24 to 1.14 of the size, and a 2px inset floor shared by the measuring and drawing code.

## 0.7.0 - 2026-11-08

- Drop one queued folder from a running batch without cancelling the whole batch: rows that have not started yet carry a dismiss button, confirm before they are removed, and the consumer skips them so the folders after them still run. The counter and the progress bar follow the folders that are still in the queue, so a batch with one folder dropped shows 3 instead of 4.
- Refuse to drop the folder that is already running, with an error telling the reader to cancel the whole batch instead: only a batch cancel can interrupt work that is on the GPU.

## 0.6.0 - 2026-11-08

- Queue several source folders as one serial batch: each folder runs the full pipeline in the order it was added, one at a time because Ollama and VRAM are exclusive, and a failing folder no longer aborts the rest of the batch. The batch reports "N succeeded / M failed" when it ends.
- Validate every queued folder before anything starts: missing paths and folders without supported images are reported together instead of surfacing mid-run, and duplicate paths (including case and trailing-separator variants) are collapsed.
- Cancel a batch as a whole: the folder in flight stops at its next progress checkpoint and queued folders never create a job at all, so no GPU work starts after the cancel.
- Release the Ollama models a finished batch was holding by evicting them with `keep_alive: 0` (what `ollama stop` does), so a completed batch does not keep VRAM resident for five minutes.
- Remember the batch's output root on the server, so reloading the page during a run restores the source list and output folder instead of silently falling back to writing beside each source.
- Show the running folder's phase on the batch card: the headline names the folder being processed and the subtitle mirrors the single-job card with "recognising page X/Y (OCR)", "translating page X/Y", or "rendering page X/Y". A failed folder keeps its error in red after the batch ends.
- Add an input language setting (auto, English, Japanese) so the OCR can be told which script to expect; declaring English skips the Japanese recogniser entirely and stops it from inventing Japanese readings for Latin artwork.
- Let the user cap the CPU thread budget, with the detected CPU model and every thread count from the logical core count down to one offered as a choice.
- Raise the Ollama context and output ceilings to 64K and scale the request timeout with the output budget, fixing the truncation at 2048 tokens that aborted structured translation and the hard 180 s timeout that killed long pages.
- Announce finished and failed jobs with two distinct chimes, with an opt-out per outcome and an adjustable volume.
- Preserve page-spanning display artwork on covers and inner titles instead of erasing it and stamping a hallucinated date on top.
- Reject text-less OCR boxes whose reading is not reproducible across crops, removing hallucinated vertical text from pages that have none.
- Recover the outer column of multi-column vertical bubbles that the detector misses, together with the rest of the group.
- Fix fast-mode rendering, inner-title artwork handling, and the erase fill colour that left white patches on tinted paper.

## 0.5.2 - 2026-07-20

- Keep full-page 9B vision results audit-only unless their text agrees with the detector-owned region, preventing neighbouring numbered boxes from swapping dialogue.
- Add exact one-crop-per-unit 9B role auditing for dialogue, sound effects, artwork text, and noise; Japanese grammar can override a wrong VLM role label.
- Preserve clothing prints, logos, props, and implausibly oversized detector regions instead of allowing them into destructive cleanup.
- Reject nested recovery duplicates both within a missing-text recovery batch and across final effect units, while allowing strong exact-crop evidence to recover an automatically skipped effect.
- Add a general Japanese manga SFX layer for common heartbeat, rumble, impact, mechanical, vocal, movement, and wet-sound families; use the 9B model only for unknown effects and never as geometry authority.
- Translate sound effects by default in the CLI, API, UI defaults, and pipeline while retaining the explicit preserve option.
- Fix JMnedict female-name classification and expand natural simplified-Chinese candidate ranking so recurring names resolve automatically without title-specific name tables.
- Remove detached orthogonal scene/artwork boxes from legacy erase groups and constrain LaMa cleanup to the remaining detector-owned text geometry.
- Treat full-page VLM omissions as evidence rather than geometry, collapse substring/containment echoes against known OCR, and re-detect only the remaining local crops.
- Preserve page-spanning decorative sound effects instead of generating destructive cleanup masks; ordinary detector-confirmed dialogue and effects remain translatable.
- Refresh inferred recurring names through natural Chinese-name candidate filtering and migrate existing transcript references without any title-specific cast table.
- Prefer high-confidence Latin logo evidence over conflicting low-confidence Japanese crop hallucinations, protecting garment and prop lettering during legacy transcript audits.
- Remove the redundant second full-page VLM recovery pass after local crop verification to reduce quality-mode latency.
- Invalidate earlier OCR caches because the semantic role and recovery contracts changed.

## 0.5.1 - 2026-07-19

- Remove thick contrasting source outlines as part of the LaMa mask instead of selecting only the dark glyph core, eliminating white silhouettes behind translated text.
- Scale the bounded cleanup halo from the measured source font size so OCR boxes that omit ruby or the outer stroke cannot leave residual Japanese fragments between detector rectangles.
- Keep grouped detector regions independent and restore every pixel outside declared text regions; the 125-page lossless WebP benchmark again passes with zero unresolved units and zero out-of-boundary pixel changes.

## 0.5.0 - 2026-07-19

- Add a fully local quality ensemble: one unrestricted Qwen3.5 9B model performs staged drafting, candidate selection, and targeted semantic-risk retranslation, with Hy-MT2 providing the independent candidate.
- Bound the 9B reviewer to a 32K working context and deterministic structured output; the unrestricted model's hidden-thinking mode could consume 32K output tokens without emitting JSON, so semantic checks are expressed directly in the verifier prompt instead.
- Add ModelScope-first automatic name resolution with JMnedict-backed recurring-name consistency, without book-specific name tables.
- Combine Paddle exact geometry, Manga OCR recognition, and full-page 9B semantic correction; VLM-only boxes can never erase artwork.
- Recover VLM-reported omissions through enlarged Paddle re-detection and independent Manga OCR confirmation, including pages where the first detector pass finds no text.
- Prevent complete hiragana dialogue from being misclassified as sound effects while preserving explicit katakana effects, isolated glyphs, and short vocalizations.
- Make six previous pages the default story context and add source-density layout budgets for compact Chinese text in narrow manga regions.
- Require only Hy-MT2 plus the selected 9B-or-smaller Ollama model in quality-mode readiness checks, and retain the single-model fast profile for lighter hardware.
- Keep lossless WebP at native pixel dimensions and preserve LaMa's strict masked-edit boundary.
- Load Hy-MT2 only for its candidate stage and explicitly release it and the 9B model between stages so LaMa rendering does not inherit translation VRAM pressure.
- Enforce one shared translation acceptance gate across draft, repair, candidate selection, and semantic audit, including simplified-Chinese normalization, source-information density, glossary consistency, and negation preservation.
- Keep ordinary A/B selection deterministic and non-thinking, reserving bounded reasoning for semantic-risk and unresolved lines instead of applying it to hundreds of ordinary wording differences.
- Filter duplicate ruby, crop echoes, malformed tiny OCR fragments, and VLM-only hallucinations before translation; exact detector geometry remains the only authority allowed to erase pixels.
- Add targeted semantic checks for questions, negation, invitations, names, laughter, and game-related actions, then retranslate only the risky units with the same local 9B model.
- Preserve large multi-column outlined lettering as one composition while retaining the original per-column erase boxes, font weight, foreground color, and contrasting stroke.
- Complete a 125-page native-resolution benchmark with 1,285 text units, 687 Chinese replacements, 598 intentionally preserved sound-effect units, zero unresolved or invalid translations, and zero changed pixels outside declared edit regions.

## 0.4.6 - 2026-07-19

- Add a distinct final repair pass when a local translation remains empty, malformed, or contains Japanese kana, while keeping the incomplete-work quality gate strict.
- Infer a compact glossary for recurring katakana character names and reject per-line candidates that do not use the selected fixed name, keeping character references coherent without an LLM API.
- Re-read sparse full-page display lettering as one MangaOCR title region, preserving the original erase boxes while restoring a single large outlined Chinese composition.
- Reject duplicated cross-unit candidates, disproportionate short-text expansions, slash-joined context, unrelated glossary names, and source-less Latin characters before they reach rendering.
- Add a deterministic local fallback for short failed vocalizations so hearts and punctuation survive without retaining kana or requiring an API.
- Persist every candidate in `translation-draft.json` so a failed translation remains diagnosable and reviewable instead of discarding model output.
- Resume failed jobs from source- and backend-validated OCR caches, including safe compatibility with built-in quality caches created by earlier releases.

## 0.4.5 - 2026-07-19

- Reuse an already-running local workspace instead of starting a second server that immediately fails on port 8765.
- Keep the Windows launcher open with an actionable message when startup genuinely fails.

## 0.4.4 - 2026-07-19

- Suppress Paddle's optional ccache notice across the actual model-construction boundary while preserving all actionable inference warnings.
- Keep Hugging Face optional-file HTTP probes out of the user console by raising only the third-party HTTP client log threshold; download failures still surface as exceptions.

## 0.4.3 - 2026-07-19

- Pin the verified Torch 2.8.0 and Torchvision 0.23.0 pair for CPU and CUDA 12.9 instead of relying on ambiguous index metadata.
- Make bootstrap diagnostics import and validate the ML runtime, so an incompatible Torchvision installation can no longer be reported as ready.

## 0.4.2 - 2026-07-18

- Allow the Torchvision 0.23 build paired with Torch 2.8 on PyTorch's CUDA 12.9 Windows index, keeping the hardware-aware overlay consistent with project metadata.

## 0.4.1 - 2026-07-18

- Keep source previews stable during job polling and request translated previews only after the selected page has actually rendered, eliminating expected 404 request spam.
- Fit tall manga pages inside the preview pane without clipping and let the canvas consume the height already created by the settings inspector.
- Show phase-aware output placeholders while OCR, coherent translation, or rendering is still in progress.
- Install Torch and Torchvision together from the selected CPU/CUDA wheel source so MangaOCR has its intended image-processing backend.
- Suppress Paddle's optional ccache notice during ordinary inference while preserving actionable runtime warnings.

## 0.4.0 - 2026-07-18

- Replace destructive text-box cleanup in quality mode with pinned Big-LaMa inpainting; inference stays at native geometry and restores every unmasked pixel exactly.
- Detect display lettering, vertical multi-column text, font scale, weight, foreground, and contrasting stroke from source pixels without a multimodal LLM.
- Add managed Noto Sans CJK SC Bold and remove the former 92 px vertical-font ceiling, allowing large outlined Chinese titles to match the source composition.
- Expand cleanup around detected regions to include furigana and thick source outlines while keeping the declared edit boundary bounded and verifiable.
- Add a conservative light-on-dark title candidate pass so reversed manga lettering is not silently missed by the default detector; MangaOCR must still confirm Japanese text.
- Add Ollama as an optional local vision-OCR backend as well as a translation backend; OCR and translation model choices are independent.
- Default to lossless WebP to reduce PNG size inflation, while retaining lossless PNG as a compatibility option.
- Add checksum-validated, atomic LaMa weight download and Apache-2.0 model attribution.
- Add generic synthetic regressions for outlined display text and white speech bubbles plus a TorchScript boundary test proving that LaMa cannot alter unmasked pixels.

## 0.3.1 - 2026-07-18

- Separate explicit reviewer skips from unresolved text; legacy `skip: true` entries no longer pass as completed work.
- Refuse to render or verify a book while any detected text unit remains unresolved, including non-empty "translations" that still contain Japanese kana or leaked prompt context.
- Record translated, explicitly skipped, preserved sound-effect, and unresolved counts in manifests and QA reports.
- Decouple OCR quality from cleanup aggressiveness; quality OCR now uses edge-aware artwork cleanup instead of destructive full-box erasure.
- Re-audit the 125-page regression book: 948 valid Chinese replacements, zero unresolved or kana-bearing translation fields, and zero pixels changed outside declared text regions.

## 0.3.0 - 2026-07-18

- Restore full PaddleOCR line-level recognition as the default quality profile; retain detection-only OCR as an explicit fast preview.
- Stop treating rotated text regions as sound effects, which previously caused skipped dialogue.
- Relax the translation length gate so natural Chinese and omitted Japanese subjects are not fragmented.
- Translate sound effects by default while keeping preservation available as an option.
- Add reviewed-transcript import for reproducible human-in-the-loop correction and rerendering.
- Add complete text-region cleanup for quality mode while keeping conservative artwork cleanup in fast mode.
- Add a full-book verifier for exact dimensions and zero pixel changes outside declared text regions.

## 0.2.0 - 2026-07-18

- Add selectable built-in Hy-MT2, local Ollama, and online OpenAI-compatible translation backends.
- Keep coherent multi-page prompting and stable text-unit mapping across every backend.
- Make Hy-MT2 optional when Ollama or an online API is selected while keeping OCR local.
- Add backend connection checks and model discovery to the UI.
- Keep online API keys in process memory or `MLS_ONLINE_API_KEY`, never in settings or job files.
- Open the UI before downloading model weights so first-time users can choose a backend.
- Avoid Windows cuDNN DLL conflicts by pairing CUDA PyTorch with CPU Paddle detection.
- Select Manga OCR's CUDA device independently so CPU Paddle does not disable GPU recognition.
- Disable the unsupported Windows CPU oneDNN/PIR path in PaddleOCR.
- Preserve panel art between grouped text lines by erasing only padded, tight detector boxes and retaining edge-connected artwork.
- Skip redundant Paddle recognition because Manga OCR already owns Japanese recognition, substantially reducing CPU batch time.
- Reject context leakage, overlong text, and untranslated kana, then retry the affected unit with a strict bounded prompt.

## 0.1.2 - 2026-07-18

- Make uv the primary environment and dependency manager on Windows and Linux.
- Commit `uv.lock` for reproducible application, ML, and test dependencies.
- Keep CPU/CUDA-specific Torch and Paddle wheels as a hardware-aware uv overlay.
- Run the full Windows/Linux and Python 3.11/3.12 CI matrix through official setup-uv.

## 0.1.1 - 2026-07-18

- Automatically provision a pinned OFL CJK font on minimal Windows/Linux systems.
- Validate font downloads before atomic installation.
- Update GitHub Actions to Node 24-based action releases.
- Normalize CLI output to UTF-8 on Windows runners and legacy code pages.

## 0.1.0 - 2026-07-18

- Initial local-first OCR, context-aware translation, and bounded rendering pipeline.
- Responsive browser workspace and CLI.
- ModelScope-first model manager with transparent Manga OCR fallback.
- One-command virtual environment and weight bootstrap for Windows and Linux.
- MIT licensing, model notices, CI, tests, design contract, and UI parity ledger.
