import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
WEB = ROOT / "src" / "manga_localizer" / "web"


def test_navigation_and_pipeline_contract():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert html.count('class="nav-item') == 4
    assert html.count('class="pipeline-step" data-phase=') == 4
    assert html.count("data-preview=") == 3
    assert 'id="prevPage" aria-label="上一页"' in html
    assert 'data-i18n-aria-label="preview.previous" disabled' in html
    assert 'id="nextPage" aria-label="下一页"' in html
    assert 'data-i18n-aria-label="preview.next" disabled' in html
    assert 'id="inferenceBackend"' in html
    assert 'id="inferenceBackendSetting"' in html
    assert 'id="ocrBackend"' in html
    assert 'id="ollamaOcrModel"' in html
    assert 'id="checkInference"' in html
    assert 'id="glossaryText"' not in html
    assert "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q6_K:latest" in html


def test_ocr_phase_uses_zero_based_dom_index():
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    assert 'if(job?.phase==="ocr")return' in javascript
    assert ">=.5?1:0" in javascript
    assert "const active=phaseIndex(job)" in javascript
    assert "state.activeBackend!==`${backend}:${ocr}`" in javascript
    assert 'textContent=t("settings.inference.notChecked")' in javascript


def test_ui_supports_persistent_chinese_japanese_and_english_locales():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    i18n = (WEB / "i18n.js").read_text(encoding="utf-8")
    assert '<script src="/i18n.js" defer></script>' in html
    assert html.index('/i18n.js') < html.index('/app.js')
    assert 'id="uiLanguage"' in html
    for locale in ("zh-CN", "ja-JP", "en-US"):
        assert f'<option value="{locale}">' in html
        assert f'"{locale}": {{' in i18n
    assert 'const STORAGE_KEY = "mls.uiLanguage"' in i18n
    assert "localStorage.setItem(STORAGE_KEY, locale)" in i18n
    assert 'window.addEventListener("mls:localechange",refreshLocalizedUI)' in javascript


def test_target_language_values_are_stable_across_ui_locales():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert '<option value="简体中文" data-i18n="target.zhHans">' in html
    assert '<option value="繁体中文" data-i18n="target.zhHant">' in html
    assert '<option value="English" data-i18n="target.english">' in html


def test_source_language_values_are_stable_across_ui_locales():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert 'id="sourceLanguage"' in html
    assert '<option value="auto" data-i18n="source.auto">' in html
    assert '<option value="en" data-i18n="source.en">' in html
    assert '<option value="ja" data-i18n="source.ja">' in html
    css = (WEB / "app.css").read_text(encoding="utf-8")
    # The setup row gained a fourth control, so the grid must have four tracks.
    assert "grid-template-columns:1fr 1fr 140px 140px" in css


def test_every_static_translation_key_exists_in_all_locales():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    i18n = (WEB / "i18n.js").read_text(encoding="utf-8")
    keys = set(re.findall(r'data-i18n(?:-[a-z-]+)?="([^"]+)"', html))
    keys.update(re.findall(r'\bt\("([^"]+)"', javascript))
    keys.update(
        {
            "models.role.paddleocr",
            "models.role.manga-ocr",
            "models.role.lama",
            "models.role.hy-mt2",
            "models.role.translation",
            "models.role.review",
            "models.role.ocr",
            "models.license.upstream",
            "models.pulled",
            "models.notPulled",
            "models.offline",
        }
    )
    missing = [key for key in sorted(keys) if i18n.count(f'"{key}":') != 3]
    assert missing == []


def test_preview_uses_available_height_and_contains_full_page():
    css = (WEB / "app.css").read_text(encoding="utf-8")
    assert ".preview-card{min-height:568px;overflow:hidden;display:grid" in css
    assert "grid-template-rows:52px minmax(458px,1fr) 56px 40px" in css
    assert ".preview-canvas{height:auto;min-height:458px" in css
    assert (
        ".preview-pane img{display:block;width:100%;height:100%;object-fit:contain"
        in css
    )


def test_history_cards_restore_the_job_paths():
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    assert "async function openJob(jobId)" in javascript
    assert '$("#sourcePath").value=job.request.source' in javascript
    assert '$("#outputPath").value=job.request.output' in javascript
    assert "renderJob(job,{quiet:true})" in javascript
    assert '$("#refreshJobs").addEventListener("click",refreshJobsFeedback)' in javascript
    assert "history.refreshed" in javascript


def test_page_slider_scrubs_the_preview():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    assert 'id="pageSlider" type="range"' in html
    assert 'data-i18n-aria-label="preview.jump"' in html
    assert "function syncPageSlider(job)" in javascript
    assert 'slider.disabled=total<2' in javascript
    assert '$("#pageSlider").addEventListener("input"' in javascript
    assert "syncPageSlider(job)}" in javascript


def test_pause_button_toggles_between_pause_and_resume():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    css = (WEB / "app.css").read_text(encoding="utf-8")
    assert 'id="pauseButton"' in html
    assert 'data-i18n-title="job.pauseDetail"' in html
    assert "job.pauseUnsupported" not in html
    assert 'const ACTIVE_STATUSES=["queued","running","paused"]' in javascript
    assert "function syncPauseButton(job)" in javascript
    assert 'const next=await api(`/api/jobs/${job.id}/${paused?"resume":"pause"}`' in javascript
    assert '$("#pauseButton").addEventListener("click",togglePause)' in javascript
    assert "syncPauseButton(job);updateStartState()" in javascript
    assert '.execution-bar>button[data-paused="true"]' in css
    assert '.job-status[data-state="paused"] .job-dot' in css


def test_refresh_restores_the_last_job_paths():
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    # A bare renderJob() left the two path inputs empty after every reload.
    assert "async function resumeJob(){try{const data=await api(\"/api/jobs\")" in javascript
    assert "if(active){await openJob(active.id);pollJob();return}" in javascript
    assert "if(latest){await openJob(latest.id);return}" in javascript


def test_output_preview_waits_until_page_has_rendered():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    assert 'id="outputPending"' in html
    assert "function outputPreviewReady(job,page)" in javascript
    assert 'job.phase==="render"&&Number(job.current)>=page' in javascript
    assert "pending.textContent=pendingPreviewMessage(job)" in javascript
    assert "image.dataset.previewKey===key" in javascript


def test_ui_polish_keeps_ollama_tags_and_paused_state_honest():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    css = (WEB / "app.css").read_text(encoding="utf-8")
    # Ollama tags are pulled by the job itself; a dead service must not gate 开始.
    # llama.cpp rows are a separate kind: they cannot be bootstrapped (the
    # downloader only knows registry ids), so they get their own message.
    assert "function missingLocalModels(){return state.models.some(model=>isBootstrapModel(model)&&model.required&&!model.ready)}" in javascript
    assert 'function missingLlamaModels(){return state.models.some(model=>model.kind==="llama"&&model.required&&!model.ready)}' in javascript
    assert "const missingLocal=!reviewed&&missingLocalModels();const missingLlama=!reviewed&&missingLlamaModels();const missing=missingLocal||missingLlama;" in javascript
    assert 't("translation.start.missingLlamaModels")' in javascript
    # A failed job still needs the history list refreshed, and it now also
    # announces itself with the error sound before that refresh.
    assert 'else if(job.status==="failed"){announceJobEnd("failure");await refreshJobs()}' in javascript
    # The paused state gets its own chip colour and keeps its pipeline step lit.
    assert 'job.status==="paused"?"paused":""' in javascript
    assert ".status-chip.paused{" in css
    assert '(job?.status==="running"||job?.status==="paused")&&index===active' in javascript
    assert 'job?.status==="paused"?t("job.paused")' in javascript
    # The model page has its own refresh button next to 准备全部模型.
    assert 'id="refreshModelsLarge"' in html
    assert '$("#refreshModelsLarge").addEventListener("click",refreshModels)' in javascript
    assert ".utility-head .head-actions{" in css
    # Saving settings now says so, and a slow inference probe cannot be fired twice.
    assert 'toast(t("settings.saved"))' in javascript
    assert 'const token=state.inferenceToken=(state.inferenceToken||0)+1' in javascript
    assert "if(token!==state.inferenceToken)return" in javascript
    assert "finally{if(token===state.inferenceToken)button.disabled=false}" in javascript


def test_sidebar_shows_the_job_clock():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    css = (WEB / "app.css").read_text(encoding="utf-8")
    assert 'id="jobClock"' in html
    assert "function jobClock(job){" in javascript
    assert 't("job.clock.start",{time:formatClock(job.started_at)})' in javascript
    assert 't("job.clock.elapsed",{time:formatDuration(elapsed)})' in javascript
    assert 't("job.clock.eta",{time:formatDuration(elapsed/progress*(100-progress))})' in javascript
    assert 't("job.clock.total",{time:formatDuration(elapsed)})' in javascript
    assert 'const clock=$("#jobClock");if(clock)clock.textContent=jobClock(job)' in javascript
    assert ".job-clock{" in css


def test_output_root_is_no_longer_derived_per_folder():
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    html = (WEB / "index.html").read_text(encoding="utf-8")
    # The output field is a *root* now: the backend appends "<name>--已汉化" per folder, so a
    # pre-filled suffix would nest twice. The per-folder derivation endpoint is not called at all.
    assert "derive-output" not in javascript
    assert "deriveOutput" not in javascript
    assert "function taskOutputPath(source,root)" in javascript
    assert "--已汉化" in javascript
    assert 'id="outputPath"' in html


def test_batch_queue_panel_and_multi_folder_entry_points():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert 'data-testid="batch-queue"' in html
    assert 'data-testid="batch-tasks"' in html
    assert 'id="batchTasks"' in html
    assert 'id="batchSummary"' in html
    assert 'id="batchStateChip"' in html
    assert 'id="batchCancel"' in html
    # Three ways to add folders: dialog, subfolder checklist, and pasting lines.
    assert 'id="folderPicker"' in html
    assert 'id="folderBulk"' in html
    assert 'id="folderChildren"' in html
    assert 'id="folderChildrenList"' in html
    assert 'id="folderChildrenConfirm"' in html
    # The source box is a textarea holding one absolute folder path per line.
    assert re.search(r'<textarea[^>]*id="sourcePath"', html)
    assert 'data-i18n-placeholder="setup.source.placeholder"' in html
    assert 'id="sourcePicker"' not in html


def test_batch_queue_renders_rows_with_remove_and_wires_its_controls():
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    assert "function isBatch(job){return !!job&&Array.isArray(job.tasks)}" in javascript
    assert "function parseQueueText(text)" in javascript
    assert "function addQueuePaths(paths)" in javascript
    assert "function batchTaskRow(task,index,batch)" in javascript
    assert 'data-index="${index}"' in javascript
    assert 'data-remove="${index}"' in javascript
    assert "function batchQueueClick(event)" in javascript
    assert 'const remove=event.target.closest("[data-remove]")' in javascript
    assert "function removeQueueAt(index)" in javascript
    assert "function selectTask(index)" in javascript
    assert "function pickChildFolders()" in javascript
    assert "function renderChildOptions(parent)" in javascript
    assert "function confirmChildFolders()" in javascript
    assert "const added=addQueuePaths(picked)" in javascript
    assert '$("#batchTasks").addEventListener("click",batchQueueClick)' in javascript
    assert '$("#batchCancel").addEventListener("click",cancelBatch)' in javascript
    assert '$("#folderBulk").addEventListener("click",pickChildFolders)' in javascript
    assert '$("#folderChildrenConfirm").addEventListener("click",confirmChildFolders)' in javascript
    assert '$("#sourcePath").addEventListener("input",()=>applySourceText(false))' in javascript
    # Browsers only expose relative names through directory pickers, and the backend needs an
    # absolute path it can open, so the UI explains why it uses the native dialog instead.
    assert "webkitdirectory" in javascript
    assert "showDirectoryPicker" in javascript


def test_batch_submission_goes_through_the_batches_endpoint_with_validation():
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    assert 'await api("/api/batches",{method:"POST"' in javascript
    assert "output:root,...collectSettings()" in javascript
    assert "function showSetupError(message)" in javascript
    assert 'showSetupError(t("batch.empty"))' in javascript
    assert 'showSetupError(t("batch.outputConflict",{path:clash}))' in javascript
    assert "!state.queue.length||missing||missingName||running" in javascript


def test_batch_cancel_stops_pending_rows_and_reuses_the_cancel_endpoint():
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    assert "function cancelBatch()" in javascript
    assert 'await api(`/api/jobs/${batch.id}/cancel`,{method:"POST"})' in javascript
    assert 'task.status="cancelled"' in javascript
    assert 'if(isBatch(job)){await cancelBatch();return}' in javascript
    assert 'window.confirm(t("batch.cancelConfirm"))' in javascript
    # Pause and resume keep targeting the current child job behind the batch id.
    assert "function jobPaused(job)" in javascript
    assert "const paused=jobPaused(job)" in javascript


def test_refresh_restores_an_active_batch_from_the_server():
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    assert 'const data=await api("/api/batches")' in javascript
    assert 'batches.find(batch=>batch.status==="queued"||batch.status==="running")' in javascript
    assert "function resumeBatch()" in javascript
    assert "const resumed=await resumeBatch();if(!resumed)await resumeJob()" in javascript
    assert "function renderBatchJob(batch,{quiet=false}={})" in javascript
    assert "if(Array.isArray(job.request.source))job.request.source=job.request.source.join" in javascript


def test_resume_restores_the_setup_form_from_the_batch_request():
    # A mid-run refresh used to restore the queue rows but leave #outputPath empty, so the
    # next submitted batch wrote beside each source folder instead of to the chosen root.
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    assert "function restoreJobForm(job)" in javascript
    assert '$("#sourcePath").value=job.request.source' in javascript
    assert '$("#outputPath").value=job.request.output' in javascript
    assert "state.queue=parseQueueText(job.request.source)" in javascript
    restored = javascript[javascript.index("function restoreJobForm(job)"):]
    restored = restored[: restored.index("async function openJob(jobId)")]
    # Legacy records created before the backend carried `request` fall back to their tasks.
    assert "if(!job.request)job.request={}" in restored
    assert 'if(job.request.source===undefined)job.request.source=(job.tasks||[]).map(task=>task.source).join("\\n")' in restored
    assert 'if(job.request.output===undefined||job.request.output===null)job.request.output=""' in restored
    assert "writeQueueText();clearSetupError();renderBatchQueue()" in restored
    # Both entry points must share the helper so they cannot drift apart again.
    assert javascript.count("restoreJobForm(") == 3
    active_branch = javascript[javascript.index('if(active){state.taskIndex=-1;restoreJobForm(active)'):]
    assert active_branch.startswith('if(active){state.taskIndex=-1;restoreJobForm(active);renderJob(active);pollJob()}')


def test_preview_follows_the_selected_batch_task():
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    assert "function previewTarget()" in javascript
    assert "function previewTotal()" in javascript
    assert "if(!task||!task.job_id)return null" in javascript
    assert "const job=previewTarget();if(!job||!job.total){resetPreview();return}" in javascript
    assert "function resetPreview()" in javascript
    assert "previewTotal()||1" in javascript


def test_batch_terminal_state_rings_the_chime_once():
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    batch_branch = javascript[javascript.index("if(isBatch(job)){"):]
    assert 'if(job.status==="completed"){announceJobEnd("complete")' in batch_branch
    assert 'else if(job.status==="failed"){announceJobEnd("failure")' in batch_branch
    assert 'else if(job.status==="cancelled"){announceJobEnd("failure")' in batch_branch
    # A finished batch is not polled again: only live batches reschedule.
    assert "else state.pollTimer=setTimeout(pollJob,1000);return}" in batch_branch


def test_batch_card_shows_the_running_folder_phase():
    # The batch card used to say only "processing 1/3", which never told the user how far the
    # current book had got. It now mirrors the single-job card: folder headline plus the
    # running child's own phase and page counters.
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    i18n = (WEB / "i18n.js").read_text(encoding="utf-8")
    css = (WEB / "app.css").read_text(encoding="utf-8")
    assert "function batchDetailText(batch)" in javascript
    assert "function batchTaskHeadline(batch)" in javascript
    assert "function currentBatchTask(batch)" in javascript
    assert "batch.current_task" in javascript
    assert "batch.total_batch_size" in javascript
    # The subtitle maps the live child's phase onto page counters.
    detail = javascript[javascript.index("function batchDetailText(batch)"):]
    detail = detail[: detail.index("function setJobDetail(")]
    for phase in ("ocr", "translate", "render"):
        assert f't("batch.task.{phase}"' in detail, phase
        assert f'task.phase==="{phase}"' in detail, phase
    assert 't("batch.task.failure",{message:' in detail
    assert 't("job.preparing")' in detail
    assert 't("job.queued")' in detail
    # Between two folders the backend still points at the finished one; that must not read as
    # "batch complete" nor as a completed subtitle.
    assert 'return task.status==="running"||task.status==="paused"||task.status==="failed"?task:null' in javascript
    assert 'const task=currentBatchTask(batch)' in detail
    assert '"complete")return t("job.complete")' in detail
    # The headline names the folder being processed.
    headline = javascript[javascript.index("function batchTaskHeadline(batch)"):]
    headline = headline[: headline.index("function batchDetailIsError(")]
    assert 't("batch.task.headline",{index:' in headline
    assert "return batchDisplayMessage(batch)" in headline
    # The failure line is red and the class cannot leak back to a single-job card.
    assert 't("batch.task.failure"' in javascript
    assert "function batchDetailIsError(batch)" in javascript
    assert 'detail.classList.toggle("error",!!error)' in javascript
    assert "setJobDetail(batchDetailText(batch),{error:batchDetailIsError(batch)})" in javascript
    assert "setJobDetail(job.error||" in javascript
    assert 'setJobDetail(t("job.selectFolders"))' in javascript
    assert ".execution-copy small.error{color:var(--danger)}" in css
    for key in ("batch.task.headline", "batch.task.ocr", "batch.task.translate", "batch.task.render", "batch.task.failure"):
        assert i18n.count(f'"{key}":') == 3, key


def test_batch_queue_can_drop_one_waiting_folder():
    # A submitted batch used to be all-or-nothing: the only escape was cancelling the whole run.
    # The ✕ on a still-waiting row now calls the per-task cancel endpoint, and every counter has to
    # shrink to `effective_total` so the dropped row stops being waited for (4 queued -> 0/3).
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    i18n = (WEB / "i18n.js").read_text(encoding="utf-8")
    css = (WEB / "app.css").read_text(encoding="utf-8")
    assert "async function dropQueuedTask(index)" in javascript
    assert "dropQueuedTask(Number(drop.dataset.dropTask))" in javascript
    assert 'const drop=event.target.closest("[data-drop-task]")' in javascript
    assert 'data-drop-task="${task.index}"' in javascript
    # The row ✕ only exists for a folder that has not started, and only inside a live batch.
    assert 'const droppable=status==="queued"&&!!batch&&(batch.status==="running"||batch.status==="queued")' in javascript
    assert 'const removable=status==="queued"&&!batch' in javascript
    assert 'window.confirm(t("batch.task.dropConfirm",{name}))' in javascript
    # The endpoint addresses the folder by its own `index`, never by the array position.
    drop = javascript[javascript.index("async function dropQueuedTask(index)"):]
    drop = drop[: drop.index("async function renderChildOptions(parent)")]
    assert "/api/batches/${batch.id}/tasks/${index}/cancel" in drop
    assert 'method:"POST"' in drop
    assert "const task=tasks.find(item=>item.index===index)" in drop
    assert "renderJob(view)" in drop
    # Dropped folders leave the denominator, and `finished` (completed + failed) is the numerator.
    assert "function batchEffectiveTotal(batch)" in javascript
    assert "if(Number.isFinite(batch.effective_total))return batch.effective_total" in javascript
    assert "function batchFinishedTotal(batch)" in javascript
    assert "return Number.isFinite(batch.finished)?batch.finished:batchFinishedCount(batch)" in javascript
    assert "total:batchEffectiveTotal(batch)" in javascript
    assert "const total=batchEffectiveTotal(batch)" in javascript
    assert "`${batchFinishedTotal(batch)} / ${batchEffectiveTotal(batch)}`" in javascript
    assert "Number.isFinite(batch.queue_position)?batch.queue_position" in javascript
    assert 'total:batchEffectiveTotal(job),failed:job.failed||0' in javascript
    # A dropped row stays visible, greyed out, and is no longer offered a ✕.
    assert 'status==="cancelled"?" cancelled":""' in javascript
    assert ".batch-task.cancelled{" in css
    assert ".batch-task-drop{" in css
    for key in ("batch.task.drop", "batch.task.dropConfirm", "batch.task.dropped"):
        assert i18n.count(f'"{key}":') == 3, key


def test_batch_translations_cover_every_locale():
    i18n = (WEB / "i18n.js").read_text(encoding="utf-8")
    keys = [
        "batch.title",
        "batch.empty",
        "batch.add",
        "batch.addChildren",
        "batch.remove",
        "batch.cancel",
        "batch.cancelConfirm",
        "batch.sourceHint",
        "batch.outputRoot.hint",
        "batch.duplicates",
        "batch.outputConflict",
        "batch.running",
        "batch.detail",
        "batch.sidebarDetail",
        "batch.summary.count",
        "batch.summary.running",
        "batch.summary.result",
        "batch.summary.cancelled",
        "batch.pickChildren.title",
        "batch.pickChildren.detail",
        "batch.pickChildren.empty",
        "batch.pickChildren.images",
        "batch.pickChildren.none",
        "batch.pickChildren.confirm",
        "batch.pickChildren.cancel",
        "batch.task.headline",
        "batch.task.ocr",
        "batch.task.translate",
        "batch.task.render",
        "batch.task.failure",
        "batch.task.drop",
        "batch.task.dropConfirm",
        "batch.task.dropped",
    ]
    keys += [f"batch.status.{status}" for status in ("queued", "running", "completed", "failed", "cancelled")]
    keys += [f"batch.task.{status}" for status in ("queued", "running", "paused", "complete", "failed", "cancelled")]
    for key in keys:
        assert i18n.count(f'"{key}":') == 3, key


def test_batch_queue_styles_reuse_the_design_variables():
    css = (WEB / "app.css").read_text(encoding="utf-8")
    assert "grid-template-columns:1fr 1fr 140px 140px" in css
    assert ".path-field.multiline{" in css
    assert ".setup-error{" in css
    assert ".batch-task.running{" in css
    assert ".batch-task.selected{" in css
    assert ".batch-task-error{" in css
    assert ".child-option.disabled{" in css


def test_llama_backend_option_and_fields():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    i18n = (WEB / "i18n.js").read_text(encoding="utf-8")
    # Both backend pickers name llama.cpp; the workspace one runs jobs and the
    # settings one only persists, so they carry separate translations.
    assert 'data-i18n="translation.inference.llama"' in html
    assert 'data-i18n="settings.inference.llama"' in html
    # Two inference pickers plus the two OCR pickers: llama.cpp can drive either
    # stage, and the OCR one is what keeps a hybrid run off the Ollama daemon.
    assert html.count('<option value="llama"') == 4
    assert 'data-i18n="translation.ocr.llama"' in html
    assert 'data-i18n="settings.ocr.llama"' in html
    # The setup block mirrors the seven backend fields the API accepts.
    assert 'id="llamaFields" class="backend-fields" hidden' in html
    for field in (
        "llamaModel",
        "llamaReviewModel",
        "llamaContextTokens",
        "llamaMainPort",
        "llamaReviewPort",
        "llamaServerPath",
        "llamaOllamaRoot",
    ):
        assert f'id="{field}"' in html, field
    assert 'id="llamaContextTokens" type="number" min="4096" max="262144"' in html
    # llama exposes its own block; Ollama's block still serves a hybrid OCR run,
    # and a llama OCR run reuses the vision-model input of Ollama's block.
    assert '$("#llamaFields").hidden=backend!=="llama"&&ocr!=="llama"' in javascript
    assert "$(\"#ollamaFields\").hidden=backend!==\"ollama\"&&!['hybrid','ollama','llama'].includes(ocr)" in javascript
    # Loading, collecting and saving all seven values keeps the form honest.
    assert '$("#llamaModel").value=settings.llama_model||""' in javascript
    assert '$("#llamaContextTokens").value=String(settings.llama_context_tokens||65536)' in javascript
    assert 'llama_context_tokens:Number($("#llamaContextTokens").value||65536)' in javascript
    assert 'llama_main_port:Number($("#llamaMainPort").value||8080)' in javascript
    assert 'llama_review_port:Number($("#llamaReviewPort").value||8081)' in javascript
    assert 'llama_server_path:$("#llamaServerPath").value.trim()' in javascript
    assert 'llama_ollama_root:$("#llamaOllamaRoot").value.trim()' in javascript
    assert '"#llamaModel","#llamaReviewModel","#llamaContextTokens","#llamaMainPort","#llamaReviewPort","#llamaServerPath","#llamaOllamaRoot"' in javascript
    # llama needs its own translation model name instead of the Ollama tag.
    assert '(backend==="llama"&&!$("#llamaModel")?.value.trim())' in javascript
    # Model rows of kind "llama" are GGUF files already on disk.
    assert 'id.startsWith("llama:")?"LC"' in javascript
    assert 'model.kind==="ollama"||model.kind==="llama"?t("models.license.upstream")' in javascript
    assert 'if(model.kind==="llama")return t(model.status_code==="offline"' in javascript
    # ...so they never join the downloader, which only knows registry ids.
    assert 'function isBootstrapModel(model){return isLocalModel(model)&&model.kind!=="llama"}' in javascript
    assert "async function bootstrapModels(){const missing=state.models.filter(model=>isBootstrapModel(model)&&model.required&&!model.ready)" in javascript
    assert "const missing=state.models.filter(model=>isBootstrapModel(model)&&model.required&&!model.ready);$(\"#bootstrapModels\").hidden" in javascript
    keys = [
        "translation.inference.llama",
        "settings.inference.llama",
        "settings.llama.title",
        "settings.llama.detail",
        "settings.llama.model",
        "settings.llama.reviewModel",
        "settings.llama.context",
        "settings.llama.mainPort",
        "settings.llama.reviewPort",
        "settings.llama.serverPath",
        "settings.llama.serverPath.placeholder",
        "settings.llama.ollamaRoot",
        "settings.llama.ollamaRoot.placeholder",
        "translation.start.missingLlamaModels",
    ]
    for key in keys:
        assert i18n.count(f'"{key}":') == 3, key


def test_cpu_thread_picker_is_built_from_the_detected_cpu():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    css = (WEB / "app.css").read_text(encoding="utf-8")
    assert 'id="cpuThreads"' in html
    assert 'data-i18n="settings.cpuThreads"' in html
    # Options are not hard-coded: the panel renders whatever /api/system found.
    assert "function renderCpuChoices(cpu,selected){" in javascript
    assert "(cpu&&cpu.thread_choices)||[]" in javascript
    assert 'cpu_threads:Number($("#cpuThreads").value||0)' in javascript
    assert '"#cpuThreads"' in javascript
    assert ".settings-form .field-hint{" in css


def test_notification_sounds_are_separate_opt_in_switches():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    css = (WEB / "app.css").read_text(encoding="utf-8")
    # One switch per event, so a user who only cares about failures can silence
    # the success chime without losing the one they actually wait for.
    assert 'id="notifyComplete"' in html
    assert 'id="notifyFailure"' in html
    assert 'data-i18n="settings.sound.complete"' in html
    assert 'data-i18n="settings.sound.failure"' in html
    # The two events must not share a timbre: completion is a rising major
    # arpeggio on a bell-like overtone stack, failure a falling minor second on a
    # square wave.  Both are brighter than the plain sine the chime started as,
    # which is why the completion cue carries harmonics instead of a waveform name.
    assert "BELL_PARTIALS=[1,0.8,0.6,0.4,0.25,0.15]" in javascript
    assert 'complete:{partials:BELL_PARTIALS,' in javascript
    assert 'failure:{type:"square"' in javascript
    # The partials must go through the two-argument form, which normalises the
    # curve to full scale: without that the overtones would add to the sample
    # peak and the amplitude headroom the volume ceiling relies on would be gone.
    assert "createPeriodicWave(real,imag)" in javascript
    assert "setPeriodicWave(soundWave(soundContext,cue.partials))" in javascript
    assert '"#notifyComplete","#notifyFailure"' in javascript
    assert "notify_on_complete:$(\"#notifyComplete\").checked" in javascript
    assert "notify_on_failure:$(\"#notifyFailure\").checked" in javascript
    assert 'announceJobEnd("complete")' in javascript
    assert 'announceJobEnd("failure")' in javascript
    # Both sounds are auditable without running a book.
    assert 'id="previewComplete"' in html
    assert 'id="previewFailure"' in html
    assert 'playCue("complete")' in javascript
    assert 'playCue("failure")' in javascript
    assert ".settings-form .sound-preview-row{" in css


def test_notification_volume_scales_the_cues_and_survives_reload():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    css = (WEB / "app.css").read_text(encoding="utf-8")
    assert 'id="notifyVolume"' in html
    assert 'data-i18n="settings.sound.volume"' in html
    assert 'id="notifyVolumeLabel"' in html
    # 0 must mean silence rather than a zero-length ramp: an exponential ramp to
    # 0 throws, so the cue is skipped before any oscillator is built.
    assert "const volume=notifyVolumeValue();if(!(volume>0))return;" in javascript
    # The peak gain is scaled by the slider, and the label tracks it live.
    assert "(cue.gains[index]||0.16)*volume" in javascript
    assert "function renderVolumeLabel(){" in javascript
    assert '$("#notifyVolume").addEventListener("input",renderVolumeLabel)' in javascript
    # Saved like every other setting, and restored from the server on load.
    assert 'notify_volume:Number($("#notifyVolume").value||0)' in javascript
    assert '"#notifyVolume"' in javascript
    assert '$("#notifyVolume").value=String(' in javascript
    assert ".settings-form .sound-volume-row{" in css


def _cue_envelope(gain, elapsed, duration):
    """The gain playCue() applies ``elapsed`` seconds into a note."""
    if elapsed < 0 or elapsed > duration:
        return 0.0
    if elapsed <= 0.02:
        return gain * (1e-4 / gain) ** (elapsed / 0.02)
    span = duration * 0.7 - 0.02
    return gain * (1e-4 / gain) ** ((elapsed - 0.02) / span)


def _cue_worst_case_peak(gains, notes, duration):
    """Upper bound on a cue's peak sample level.

    Every waveform a cue uses -- the built-in types and the two-argument
    createPeriodicWave alike -- is normalised to +/-1 peak, so a note's level is
    its gain times its envelope.  The notes are staggered and each decays while
    the next rises, so sampling the total at each note's own peak instant bounds
    the peak far more tightly than assuming every note peaks together.
    """
    worst = 0.0
    for _, offset in notes:
        instant = offset + 0.02
        level = sum(
            _cue_envelope(gain, instant - other, duration)
            for gain, (_, other) in zip(gains, notes)
        )
        worst = max(worst, level)
    return worst


def test_notification_cues_keep_amplitude_headroom():
    """The cue ceiling may only rise while the notes still fit in full scale."""
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    assert "createPeriodicWave(real,imag)" in javascript
    cues = re.search(r"const SOUND_CUES=\{(.*?)\};", javascript, re.S).group(1)
    parsed = [
        (
            [float(value) for value in re.findall(r"[\d.]+", gain_group)],
            [[float(frequency), float(offset)] for frequency, offset in re.findall(r"\[([\d.]+),([\d.]+)\]", note_group)],
            float(duration),
        )
        for gain_group, note_group, duration in re.findall(
            r"gains:\[([^\]]*)\],notes:\[(.*?)\],duration:([\d.]+)", cues
        )
    ]
    assert len(parsed) == 2
    # Past full scale the browser output clips on the loudest setting, so this is
    # the bound that decides how far the volume ceiling can be raised.
    for gains, notes, duration in parsed:
        assert len(gains) == len(notes)
        assert _cue_worst_case_peak(gains, notes, duration) <= 0.7
    # And the cue still has to be able to get loud: a table an order of magnitude
    # quieter would satisfy the bound above and be useless.
    assert min(sum(gains) for gains, _, _ in parsed) >= 0.7


def test_ml_bootstrap_installs_matching_torchvision_backend():
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    powershell = (ROOT / "scripts" / "bootstrap.ps1").read_text(encoding="utf-8")
    shell = (ROOT / "scripts" / "bootstrap.sh").read_text(encoding="utf-8")
    assert '"torch==2.8.0"' in project
    assert '"torchvision==0.23.0"' in project
    assert '"torch==2.8.0+cu129", "torchvision==0.23.0+cu129"' in powershell
    assert "'torch==2.8.0+cu129' 'torchvision==0.23.0+cu129'" in shell
    assert "doctor --require-ml" in powershell
    assert "doctor --require-ml" in shell


def test_a_failed_poll_reschedules_instead_of_ending_the_updates():
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    poll = javascript.split("async function pollJob(){", 1)[1].split("\n", 1)[0]
    # One dropped request used to end the loop for good: the job kept running
    # with the progress bar frozen where it stopped and the page never learned
    # that it had finished.
    assert "state.pollFailures=0;" in poll
    assert "state.pollFailures=(state.pollFailures||0)+1" in poll
    assert "state.pollTimer=setTimeout(pollJob,Math.min(8000," in poll
    assert "pollFailures:0" in javascript
def test_a_poll_tick_keeps_the_rows_the_user_is_about_to_click():
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    # Every tick used to rewrite the whole task list, so the row under the
    # cursor and its drop button were destroyed and rebuilt about once a second.
    assert "batchRows:null" in javascript
    assert "function paintBatchRows(rows,htmls)" in javascript
    assert "if(painted[index]===htmls[index])continue" in javascript
    assert "paintBatchRows(rows,tasks.map(" in javascript
    assert "rows.innerHTML=tasks.length?" not in javascript


def test_settings_are_saved_once_after_the_user_stops_changing_them():
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    assert "function queueSettingsSave()" in javascript
    assert 'setTimeout(saveSettings,300)' in javascript
    assert 'addEventListener("change",queueSettingsSave)' in javascript
    assert 'addEventListener("change",saveSettings)' not in javascript
    save = javascript.split("async function saveSettings(){", 1)[1].split("\n", 1)[0]
    # A settings save used to drag a full model re-discovery behind it.
    assert "refreshModels" not in save


def test_the_history_list_is_only_rebuilt_when_it_changed():
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    assert "if(html!==state.jobsHtml){state.jobsHtml=html;list.innerHTML=html}" in javascript
    # One listener on the container instead of one per row.
    assert "$$('.job-item').forEach(" not in javascript
    assert '"#jobList").addEventListener("click"' in javascript
