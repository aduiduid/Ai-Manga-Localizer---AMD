# Changelog

## 0.10.11 - 2026-11-12

- Typesetting cannot fail on a fractional box.  The growth allowance is a ratio
  (1.7x, 2.4x), so an odd-sized box grew by a fraction of a pixel; numpy slices
  truncate that silently but `PIL.paste` does not, and page 38 of a 98-page book
  ended the job at 38% with "'float' object cannot be interpreted as an integer".
  The allowance is rounded and both `_erase` and `render_page` coerce their box,
  so the geometry is integral by construction instead of by luck.
- A page whose text is Latin keeps the strings Paddle read below the Japanese
  confidence floor.  The 0.72 floor was tuned on Japanese pages, where a low
  score usually means a hallucinated reading; on an English page the recognized
  string is the evidence.  The "'SUP" bubble (43x27 px, score 0.67) was read
  correctly and dropped, so it was never translated and stayed in English beside
  the Chinese text.
- An audit of six English pages shows the lower floor only adds real text
  ("'SUP", "puMP PUMP", "S-SLOW DOWN.", "$I'm--!", "MY DICKYOU CUmE.", ".HEH")
  and removes nothing: two-line captions join up, and no page loses a unit.
- A Latin source is never typeset as columns.  The vertical rule (height at least
  0.62 of width) exists for stacks of Japanese columns, but on a nearly square
  Latin box it drew the three-character translation one character per column from
  the right edge, so the bubble read backwards ("？样咋").  Vertical layout now
  also requires Japanese in the source text.
- Page caches from earlier versions are invalidated, because a cached page still
  holds the OCR result without the recovered bubble.
- The fast (artwork) profile erases the whole line again.  It cleans one detector
  line at a time, so its search window is barely wider than the line; the 6%
  component cap introduced for the generous quality window is smaller than a
  single letter at that size, and every glyph was dismissed as artwork, leaving
  "ARRIVED", "LOOKS ADORABLE" and "ARTNER" showing through the translation.  The
  cap is now a parameter: 12% for the tight artwork window, 6% for quality.

## 0.10.10 - 2026-11-12

- Short dialogue stops being squeezed to an unreadable size.  The typesetting
  box grew one pixel at a time and re-tested its edges against the span grown so
  far, which only ever worked on a rectangle: on a balloon the box got taller
  first, the taller edge line crossed the widening curve, and growth stopped
  while there was still room inside.  A 78x22 box stalled at 83x28, no readable
  size fitted, and the overflow sent the line onto the panel outside.  The page
  is now split into same-coloured regions and the balloon is the one the box's
  rim touches; on the reported page the same text reaches 14-20px instead of
  8-13px and stays inside its balloon.
- Two balloons that touch are one region, so each box is cut back to the midline
  between it and its nearest neighbour.  Without that both boxes grew to the
  whole merged shape and the two translations were painted on top of each other.
- Cleaning no longer eats the artwork around the lettering.  The mask is clipped
  to the lettering's own band (a little wider along the text, a little shorter
  across it), so a balloon outline, a lock of hair or a hat brim inside the
  searched rectangle can no longer come back inpainted.  On the reported page
  the pixels changed outside those bands fell from 17188 to 0, and the total
  changed from 32387 to 16463, while the source text is still removed as
  thoroughly as before (residue inside the lettering: 91 -> 93 pixels).
- The searched rectangle itself stays generous on purpose: the paper branch and
  the glyph polarity are read from it, and a crop holding nothing but lettering
  makes both statistics meaningless.

## 0.10.9 - 2026-11-12

- A browser tab that vanishes mid-poll no longer prints a traceback in the
  console.  On Windows the proactor's `_call_connection_lost` calls `shutdown()`
  on a socket the client already reset, which raises `ConnectionResetError`
  (WinError 10054) and lands in the event loop's default exception handler once
  per closed tab.  The loop now drops that one exception and forwards the rest.
- The job panel no longer freezes after a single failed poll.  Every failure
  used to toast and stop rescheduling, so a momentary connection error left the
  page showing a stale percentage forever; polls now back off to at most 8 s and
  keep going.

## 0.10.8 - 2026-11-11

- A crashed llama-server is restarted under the request instead of failing the
  book: Windows logged a heap corruption (`0xc0000374` in `ntdll.dll`) that
  killed one process mid-request after 93 minutes, and the job died with it.
  Any unreachable server is now replaced and the same request is sent once more,
  within a finite restart budget.
- A run that dies late resumes its own work.  `translation-draft.json` used to
  be written and never read; the next run now loads it and only translates the
  units still missing, so re-running the 152-page book that failed at 18:41
  reused 1108 of its 1111 lines and finished the remaining 3 in 19 s.
- Servers are replaced after 30 minutes of continuous uptime.  This build leaks
  (Windows flags `llama-server.exe` with `RADAR_PRE_LEAK_64`), and the age of
  the process is the only warning available before its heap gives way.

## 0.10.7 - 2026-11-11

- A decoder that collapses into one repeated glyph no longer stalls a
  book.  The reply is recognised as a collapse, that batch is skipped
  instead of being fanned out into one model call per line, and the job
  reports how many batches and lines were left in the source language.
- Translation requests now ask for a mild repeat penalty (1.1 over the
  last 64 tokens); llama-server's own default is off.  A collapsed reply
  restarts that llama-server once, then reports instead of retrying.
- Progress is reported for every page as it is translated, not once per
  four-page chunk, and a collapsed run says so in the job message.
- Every request has a hard wait ceiling, so a silent server ends in an
  error instead of an indefinite wait.

## 0.10.6 - 2026-11-09

- One dense batch no longer loses its tail.  The batch prompt covers up to
  four pages, and the four densest pages of the English reference set held
  93 units.  Its answer was capped at 1600 tokens, so the reply stopped
  mid-list: six units fell into the single-line fallback on the first pass,
  and the independent candidate pass returned seventy empty translations
  from the same cut.  The budget now scales with the source (three tokens
  per character, up to 4096) and a chunk stops at 40 units, which keeps the
  request small enough that the answer always fits.
- The [拟声词] label the prompt asks for is no longer counted as output.
  ``_valid_sfx_translation`` measured the answer's length before stripping
  the bracket label, so a correct ``[拟声词] 叹息`` for ``SIGH`` counted five
  characters against a budget of four and was thrown away.
- An echoed story context is retried without it.  The 1.8B candidate
  answers a long context prompt by repeating the context back verbatim
  often enough to matter: one 36-unit batch came back as the context
  section itself, so none of its ids parsed and every line paid its own
  model call.  When a batch answer parses to nothing and a story context
  was supplied, the batch is asked once more without it.
- Two short cries may now share one translation.  ``HAH! AH!`` and
  ``HAH! AHI`` both read 「哈！啊！」; the duplicate guard treated a shared
  answer as one line leaking into the next and rejected both.  A source of
  ten characters or fewer with no kanji carries no context to leak.
## 0.10.5 - 2026-11-09

- Do not pay for recall proposals a Latin book can never use.  A declared
  English source has no recovery path: PaddleMangaOCR.recover_missing
  returns the page untouched for it, because the reading that confirms a
  proposal is Japanese-only.  The hybrid pass still asked the vision model
  for those proposals and built the ink tiles anyway, which cost 13-21% of
  each page for geometry that could never be written back.  Both are now
  skipped when the source language is declared en; auto keeps them.
  Measured on the five-page English reference set: OCR 1314.1 s -> 1027.6 s
  (-21.8%), whole run 1707.1 s -> 1430.3 s (-16.2%), with all 129 units and
  every per-unit verdict byte-identical.
- Batched crop audits were tried and rejected.  Sending four tight crops in
  one request cut that stage from 328 s to 84 s, but the model changed its
  own verdict on the same crop: RISE and PUMP came back artwork one at
  a time and sfx in a batch of four, so two pages lost their lettering to
  an erase-and-replace.  No threshold can recover a verdict the model no
  longer gives, so the audit keeps one crop per request.

## 0.10.4 - 2026-11-09

- A Latin aside is no longer erased from the translation: the full-page pass
  reads `SUPONGO...` as dialogue, but the tight-crop audit looked at the same
  words alone, called them artwork lettering and set `skip` -- so the bubble
  kept its Spanish while the caption beside it was translated.  The page-level
  verdict is now recorded as `page_dialogue` and the crop audit may not
  demote it.  `likely_dialogue_text` still only recognises Japanese, so the
  guard depends on the text agreeing, not on the script.
- The English regression set keeps its 129 units with no unit lost or added;
  two artwork skips become real translations (`HOW DO ILOOK?` and `MHM.`),
  which is the same bug on English pages, not new false positives.
- `OCR_CACHE_VERSION` goes to 12: pages cached by 0.10.3 still hold the
  `artwork_text` verdict and would not be re-read otherwise.


## 0.10.3 - 2026-11-09

- Dense Japanese pages keep the small handwritten sound effects that
  llama.cpp never proposes: `ink_tile_hints()` scans fixed 3x4 ink tiles and
  feeds them to the existing `recover_missing` path, whose Paddle
  re-detection and independent MangaOCR read decide what actually becomes a
  unit.  Three consecutive runs on the vertical-text page returned 12 units
  each (Ollama's count) with no lost unit.
- A decoder whose confidences do not separate text from noise no longer
  silently vetoes its own proposals: `scores_are_informative()` detects
  all-negative, saturated or flat score sets (llama.cpp answers a constant
  -0.0127 for every proposal, twice in a row) and skips the `0.72`/`0.75`
  recall gates, sending the proposals to the geometry check instead.  Ollama's
  healthy 0.71-1.00 spread keeps its gates.
- A nested effect is only dropped as a duplicate when the larger unit already
  carries the same text, so `びっ` inside `バアン` survives.
- The English regression set is unchanged: the same 129 units and 10 artwork
  skips on all five pages.
- Recommended split stays: English/Western books on llama.cpp (1 GB less VRAM,
  no resident daemon, 7% faster on the densest page), Japanese vertical text
  on Ollama (2.9x faster per page).  The recall fix brings llama.cpp to 12/12
  units on the Japanese page but two of those positions read back as `ハア`
  and `な`, and a 27x26 px mark stays undetected, so Ollama still wins on
  accuracy for that script.  Both gaps are recorded as TODO in the README
  rather than patched by lowering thresholds.


## 0.10.2 - 2026-11-09

- Recommend one backend per text direction, in the README and on the settings page: English/Western books on llama.cpp (1 GB less VRAM, no resident daemon, 7% faster on the densest page), Japanese vertical text on Ollama (2.9x faster per page, and 3 more units found). The two share the same geometry layer, so switching costs nothing.
- Add the full Ollama vs llama.cpp comparison to `docs/ollama-vs-llamacpp-ab.md`.
- Correct the README's earlier "43-45 s per page against 333 s": that was measured on a 4-unit page and does not generalise to dense pages, where both backends land in the same 250-620 s band.

## 0.10.1 - 2026-11-09

- The llama.cpp vision adapter now escalates its wait with the output budget and penalises repeats on the retry, exactly as the Ollama adapter does. A flat 300 s timeout aborted the densest page of the A/B book mid-generation, so the retry that recovers a degenerate page never ran.
- The detection prompt no longer shows `"score":0.0` in its JSON example: llama.cpp decodes against the schema as a grammar and copies a boundary value out of the example verbatim, which zeroed every box's confidence and closed the score gates downstream (`prefer_semantic_ocr` needs 0.72).
- llama.cpp vision calls wait 600 s by default, because the same refine call that finished inside the daemon's 300 s ran past it on the same page.

## 0.10.0 - 2026-11-08

- Read pages through llama.cpp as well: `ocr_backend="llama"` runs the same `HybridMangaOCR` geometry-then-semantics split, with `LlamaVisionOCR` in place of the Ollama vision adapter. Because only the transport is swapped, a page's boxes and text are directly comparable between the two backends; an A/B on the sample book matched 4/4 boxes and 4/4 readings on both pages, with identical skip and role verdicts.
- Convert pages to PNG for the vision request. The llama.cpp image loader rejects WebP, which is the only format this app stores, so every page would otherwise fail on the first image.
- Start the vision model on its own port (8082) and reuse the translation slot when it already holds the same tag. The main model is the multimodal one, so batch translation would otherwise load a second copy of the same 7 GB weights; the vision slot still gets its own port because `-np 1` serves one request at a time and OCR cannot queue behind translation.
- Bounds the confidence `score` inside the detection schema. llama.cpp compiles the JSON schema into a constrained decoding grammar, and an unbounded number let it answer `0.0` for every box on a page it had read perfectly -- which silently disabled the score gates downstream (`prefer_semantic_ocr` wants 0.72, `_local_crop_is_suspicious` rejects <= 0.05). The type is stated as `minimum: 0, maximum: 1`, exactly as `refine` and `find_missing` already stated theirs.
- Release the servers when either stage started one: a finished batch now stops the vision server even when translation stayed on the Ollama daemon.
- Add `llama` to both OCR pickers and the settings panel: a run whose OCR is on llama.cpp reuses the Ollama visual-model field for the vision tag, and the start button asks for it when either stage needs it.
- The whole pipeline now runs on llama.cpp with no Ollama daemon involved: pick `llama` for both the inference and the OCR backend and translation, review and page reading all go through `llama-server.exe`. The Ollama path is untouched and stays available as the fallback -- switching either picker back is enough. Setup is described in `README.md` under "推理后端：Ollama 与 llama.cpp": the model tags reuse the Ollama names, the context defaults to 65536, and the ports default to 8080/8081/8082.
- Give the CLI the same llama.cpp settings as the UI, and add `--llama-*` flags for one-off overrides. `--backend llama` used to ignore the saved settings and silently run on the default tags and ports; the flags now fall back to what the settings page saved, so both entry points drive the same configuration.

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
