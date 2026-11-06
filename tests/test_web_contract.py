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
    assert (
        "state.models.some(model=>isLocalModel(model)&&model.required&&!model.ready)"
        in javascript
    )
    # A failed job still needs the history list refreshed.
    assert 'else if(job.status==="failed"){await refreshJobs()}' in javascript
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


def test_derive_output_lands_under_the_picked_output_root():
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    assert 'if(target==="#outputPath"&&$("#sourcePath").value.trim())await deriveOutput()' in javascript
    assert 'body:JSON.stringify({initial:source,output:$("#outputPath").value.trim()||null})' in javascript


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
