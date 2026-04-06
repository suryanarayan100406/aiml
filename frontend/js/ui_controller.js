/**
 * UIController — Main application controller.
 * Manages UI state, session lifecycle, chart rendering, toast notifications,
 * and the ANI Flow Guardian integration.
 */
(async function () {
    'use strict';

    // ─── State ────────────────────────────────────────────────
    const state = {
        activePanel: 'dashboard',
        sessionActive: false,
        sessionStart: null,
        inferenceInterval: null,
        intervalMs: 30000,
        results: [],
        pipeline: null,
        guardian: null,
        profile: window.userProfile,
        micEnabled: false,
        screenEnabled: false,
        webcamEnabled: false,
        focusTimerInterval: null,
    };

    // ─── DOM References ───────────────────────────────────────
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    // ─── Init ─────────────────────────────────────────────────
    async function init() {
        updateLoadingStatus('Loading user profile...', 20);
        await state.profile.load();

        updateLoadingStatus('Initializing inference pipeline...', 50);
        state.pipeline = new InferencePipeline();
        const demoMode = $('#settings-demo')?.checked ?? true;
        await state.pipeline.init(demoMode);

        updateLoadingStatus('Initializing Flow Guardian...', 70);
        state.guardian = new AniGuardian();

        updateLoadingStatus('Setting up interface...', 85);
        setupNavigation();
        setupSessionControls();
        setupSettings();
        setupAudioVisualizer();
        setupGuardianControls();
        updateUserBadge();

        updateLoadingStatus('Ready!', 100);
        setTimeout(() => {
            $('#loading-overlay').classList.add('fade-out');
            $('#app').classList.remove('hidden');
            setTimeout(() => $('#loading-overlay').style.display = 'none', 600);
        }, 500);
    }

    function updateLoadingStatus(text, progress) {
        const statusEl = $('#loading-status');
        const progressEl = $('#loading-progress');
        if (statusEl) statusEl.textContent = text;
        if (progressEl) progressEl.style.width = progress + '%';
    }

    // ─── Navigation ───────────────────────────────────────────
    function setupNavigation() {
        $$('.nav-links li').forEach(item => {
            item.addEventListener('click', () => {
                const panel = item.dataset.panel;
                if (!panel) return;

                $$('.nav-links li').forEach(li => li.classList.remove('active'));
                item.classList.add('active');

                $$('.panel').forEach(p => p.classList.remove('active'));
                $(`#panel-${panel}`).classList.add('active');

                $('#panel-title').textContent = item.textContent.trim();
                state.activePanel = panel;

                if (panel === 'history') loadHistory();
            });
        });
    }

    // ─── Session Controls ─────────────────────────────────────
    function setupSessionControls() {
        $('#btn-mic').addEventListener('click', async () => {
            if (!state.micEnabled) {
                const ok = await state.pipeline.enableMicrophone();
                if (ok) {
                    state.micEnabled = true;
                    $('#btn-mic').innerHTML = '<span>🎙️</span> Microphone Active';
                    $('#btn-mic').classList.add('btn-primary');
                    $('#btn-mic').classList.remove('btn-outline');
                    showToast('🎙️', 'Microphone enabled — speak to see the audio meter react!');
                    startLiveAudioLevel(); // Start reactive audio visualizer
                } else {
                    showToast('❌', 'Microphone access denied');
                }
            }
        });

        $('#btn-screen').addEventListener('click', async () => {
            if (!state.screenEnabled) {
                const ok = await state.pipeline.enableScreenCapture();
                if (ok) {
                    state.screenEnabled = true;
                    $('#btn-screen').innerHTML = '<span>🖥️</span> Screen Sharing Active';
                    $('#btn-screen').classList.add('btn-primary');
                    $('#btn-screen').classList.remove('btn-outline');
                    showToast('🖥️', 'Screen capture enabled');
                } else {
                    showToast('❌', 'Screen capture was cancelled');
                }
            }
        });

        // Webcam button — enables camera for phone/desk detection
        $('#btn-webcam')?.addEventListener('click', async () => {
            if (!state.webcamEnabled) {
                const ok = await state.pipeline.enableWebcam();
                if (ok) {
                    state.webcamEnabled = true;
                    $('#btn-webcam').innerHTML = '<span>📷</span> Webcam Active';
                    $('#btn-webcam').classList.add('btn-primary');
                    $('#btn-webcam').classList.remove('btn-outline');
                    showToast('📷', 'Webcam enabled — point it at your desk to detect phone!');

                    // Show preview
                    const container = $('#webcam-preview-container');
                    if (container) container.classList.remove('hidden');

                    // Start webcam preview loop
                    startWebcamPreview();
                } else {
                    showToast('❌', 'Webcam access denied');
                }
            }
        });

        $('#btn-start').addEventListener('click', startSession);
        $('#btn-stop').addEventListener('click', stopSession);
    }

    /** Live webcam preview with detection overlay */
    function startWebcamPreview() {
        const canvas = $('#webcam-preview');
        if (!canvas) return;

        function render() {
            if (!state.webcamEnabled) return;

            const vp = state.pipeline.getVisionPreprocessor();
            if (vp && vp.isActive) {
                vp.drawDetections(canvas);
            }

            requestAnimationFrame(render);
        }
        requestAnimationFrame(render);
    }

    // ─── Guardian Controls ────────────────────────────────────
    function setupGuardianControls() {
        $('#btn-focus-mode')?.addEventListener('click', () => {
            startFocusMode();
        });

        $('#btn-end-focus')?.addEventListener('click', () => {
            endFocusMode();
        });
    }

    async function startSession() {
        const taskText = $('#task-input')?.value || '';
        if (!taskText.trim()) {
            showToast('⚠️', 'Please describe what you\'re working on');
            return;
        }

        state.sessionActive = true;
        state.sessionStart = Date.now();
        state.results = [];
        state.guardian.reset();

        $('#btn-start').classList.add('hidden');
        $('#btn-stop').classList.remove('hidden');

        // Switch to dashboard
        $$('.nav-links li').forEach(li => li.classList.remove('active'));
        $$('.nav-links li')[0].classList.add('active');
        $$('.panel').forEach(p => p.classList.remove('active'));
        $('#panel-dashboard').classList.add('active');
        $('#panel-title').textContent = 'Dashboard';

        showToast('▶', 'Session started');
        updateModelStatus('online', 'Models active');

        // Clear welcome message and show session start
        const msgContainer = $('#guardian-messages');
        if (msgContainer) {
            msgContainer.innerHTML = '';
            addGuardianMessage('🚀', `Session started! I'm analyzing your work on: "${taskText.substring(0, 60)}${taskText.length > 60 ? '...' : ''}" — let's find your flow!`, 'pleased');
        }

        // Run first inference immediately
        await runInference(taskText);

        // Set up periodic inference
        const interval = parseInt($('#settings-interval')?.value || '30000');
        state.intervalMs = interval;
        state.inferenceInterval = setInterval(() => runInference(taskText), interval);

        // Start audio visualizer animation
        if (state.micEnabled) startAudioVisualizerLoop();

        // Start session duration updater
        updateSessionDurationLoop();
    }

    function stopSession() {
        state.sessionActive = false;
        if (state.inferenceInterval) {
            clearInterval(state.inferenceInterval);
            state.inferenceInterval = null;
        }

        $('#btn-start').classList.remove('hidden');
        $('#btn-stop').classList.add('hidden');

        // Show session summary from guardian
        const summary = state.guardian.getSessionSummary();
        if (summary) {
            addGuardianMessage(summary.emoji, summary.message, 'pleased');
        }

        // Save session summary
        if (state.results.length > 0) {
            const avgQuality = state.results.reduce((s, r) => s + r.workQuality, 0) / state.results.length;
            const session = {
                date: new Date().toISOString(),
                duration: Math.round((Date.now() - state.sessionStart) / 1000),
                samples: state.results.length,
                averageQuality: avgQuality,
                dominantState: getDominantState(),
                task: $('#task-input')?.value || '',
                workQuality: avgQuality,
            };

            state.profile.addSession(session);
            state.profile.saveSession(session);
            state.profile.save();
        }

        state.pipeline.stop();
        updateModelStatus('offline', 'Models offline');
        showToast('⏹', 'Session ended');
        
        // Hide focus button
        const focusBtn = $('#btn-focus-mode');
        if (focusBtn) focusBtn.style.display = 'none';
    }

    function getDominantState() {
        if (state.results.length === 0) return 'UNKNOWN';
        const counts = [0, 0, 0, 0, 0];
        state.results.forEach(r => { if (r.flowState >= 0 && r.flowState < 5) counts[r.flowState]++; });
        const maxIdx = counts.indexOf(Math.max(...counts));
        return ['PSEUDO_WORKING', 'TASK_SWITCHING', 'DISTRACTED', 'SOFT_FLOW', 'DEEP_FLOW'][maxIdx];
    }

    async function runInference(taskText) {
        if (!state.sessionActive) return;

        try {
            const result = await state.pipeline.analyze(taskText);
            state.results.push(result);
            updateDashboard(result);
            updateSessionDuration();

            // Get guardian response
            const response = state.guardian.generateResponse(result);
            displayGuardianResponse(response);
        } catch (err) {
            console.error('Inference error:', err);
        }
    }

    // ─── Guardian Display ─────────────────────────────────────
    function displayGuardianResponse(response) {
        // Add message
        addGuardianMessage(response.emoji, response.message, response.mood);

        // Update quality badge
        const badge = $('#guardian-quality');
        if (badge) {
            badge.textContent = `${response.workQualityProbability}% Quality`;
            const qualityNum = parseFloat(response.workQualityProbability);
            if (qualityNum > 60) {
                badge.style.background = 'rgba(16, 185, 129, 0.15)';
                badge.style.color = '#10B981';
            } else if (qualityNum > 35) {
                badge.style.background = 'rgba(245, 158, 11, 0.15)';
                badge.style.color = '#F59E0B';
            } else {
                badge.style.background = 'rgba(239, 68, 68, 0.15)';
                badge.style.color = '#EF4444';
            }
        }

        // Update avatar mood
        const avatar = $('#guardian-avatar');
        if (avatar) {
            avatar.className = `guardian-avatar mood-${response.mood}`;
            if (response.severity === 'high') avatar.classList.add('urgent');
        }

        // Display action items as chips
        if (response.actionItems && response.actionItems.length > 0) {
            const actionsContainer = $('#guardian-actions');
            // Remove old action chips
            actionsContainer.querySelectorAll('.action-chip').forEach(c => c.remove());
            response.actionItems.forEach(action => {
                const chip = document.createElement('span');
                chip.className = 'action-chip';
                chip.textContent = action;
                actionsContainer.appendChild(chip);
            });
        }

        // Show focus button if suggested
        const focusBtn = $('#btn-focus-mode');
        if (focusBtn) {
            focusBtn.style.display = response.suggestFocus ? 'inline-flex' : 'none';
        }
    }

    function addGuardianMessage(emoji, text, mood = '') {
        const container = $('#guardian-messages');
        if (!container) return;

        const msg = document.createElement('div');
        msg.className = `guardian-message${mood ? ` mood-${mood}` : ''}`;
        msg.innerHTML = `
            <span class="guardian-msg-emoji">${emoji}</span>
            <span class="guardian-msg-text">${text}</span>
        `;
        container.appendChild(msg);

        // Keep last 10 messages
        while (container.children.length > 10) {
            container.removeChild(container.firstChild);
        }

        // Auto-scroll
        container.scrollTop = container.scrollHeight;
    }

    // ─── Focus Mode ───────────────────────────────────────────
    function startFocusMode() {
        const result = state.guardian.startFocusMode(25);
        showToast(result.emoji, result.message);
        addGuardianMessage(result.emoji, result.message, 'happy');

        // Show overlay
        $('#focus-timer-overlay')?.classList.remove('hidden');

        // Update timer display
        state.focusTimerInterval = setInterval(() => {
            if (!state.guardian.isFocusMode) {
                endFocusMode();
                return;
            }
            const timeStr = state.guardian.getFocusTimeFormatted();
            const timerEl = $('#focus-timer-value');
            if (timerEl) timerEl.textContent = timeStr;

            // Update ring progress
            const total = 25 * 60;
            const remaining = state.guardian.focusTimeRemaining;
            const progress = (1 - remaining / total) * 327; // 327 = 2πr where r=52
            const ring = $('#focus-ring-progress');
            if (ring) ring.setAttribute('stroke-dashoffset', 327 - progress);
        }, 1000);

        // Hide focus button
        const focusBtn = $('#btn-focus-mode');
        if (focusBtn) focusBtn.style.display = 'none';
    }

    function endFocusMode() {
        const result = state.guardian.endFocusMode();
        showToast(result.emoji, result.message);
        addGuardianMessage(result.emoji, result.message, 'happy');

        // Hide overlay
        $('#focus-timer-overlay')?.classList.add('hidden');

        // Clear interval
        if (state.focusTimerInterval) {
            clearInterval(state.focusTimerInterval);
            state.focusTimerInterval = null;
        }
    }

    // ─── Dashboard Updates ────────────────────────────────────
    function updateDashboard(result) {
        // Flow state circle
        const flowColors = ['#EF4444', '#F59E0B', '#F97316', '#10B981', '#8B5CF6'];
        const flowCircle = $('#flow-circle');
        if (flowCircle) {
            flowCircle.style.background = `conic-gradient(from 0deg, ${flowColors[result.flowState]} 0%, ${flowColors[result.flowState]}40 100%)`;
        }

        $('#flow-emoji').textContent = result.flowEmoji;
        $('#flow-label').textContent = result.flowLabel;
        $('#confidence-badge').textContent = `${(result.confidence * 100).toFixed(0)}% conf`;

        // Probability bars
        result.probabilities.forEach((prob, i) => {
            const fill = $(`.prob-fill[data-class="${i}"]`);
            if (fill) fill.style.width = `${prob * 100}%`;
            const vals = $$('.prob-val');
            if (vals[i]) vals[i].textContent = `${(prob * 100).toFixed(0)}%`;
        });

        // Vision metrics + data source badge
        if (result.vision) {
            $('#metric-tabs').textContent = result.vision.tabCount;
            $('#metric-phone').textContent = result.vision.phoneVisible;
            $('#metric-distractions').textContent = result.vision.distractions;
            $('#metric-focus').textContent = result.vision.focusRatio;
            const vs = $('#vision-status');
            if (vs) {
                const src = result.vision.source || 'simulated';
                if (src === 'webcam' || src === 'yolo-webcam') { vs.textContent = 'Webcam'; vs.style.cssText = 'color:#10B981'; }
                else if (src === 'extension') { vs.textContent = 'Extension'; vs.style.cssText = 'color:#10B981'; }
                else if (src === 'screen-capture') { vs.textContent = 'Screen'; vs.style.cssText = 'color:#06B6D4'; }
                else if (src === 'webcam-heuristic') { vs.textContent = 'Webcam*'; vs.style.cssText = 'color:#06B6D4'; }
                else { vs.textContent = 'Simulated'; vs.style.cssText = 'color:#F59E0B'; }
            }
        }

        // Audio metrics + data source badge
        if (result.audio) {
            $('#metric-speech').textContent = result.audio.speechClass;
            $('#metric-wpm').textContent = result.audio.wpm;
            $('#metric-fluency').textContent = result.audio.fluency;
            $('#metric-audio-conf').textContent = result.audio.confidence;
            const as = $('#audio-status');
            if (as) {
                const src = result.audio.source || 'simulated';
                if (src === 'microphone') { as.textContent = 'Microphone'; as.style.cssText = 'color:#10B981'; }
                else { as.textContent = 'Simulated'; as.style.cssText = 'color:#F59E0B'; }
            }
        }

        // NLP metrics
        if (result.nlp) {
            $('#metric-task-type').textContent = result.nlp.taskType;
            $('#metric-demand').textContent = result.nlp.demand;
            $('#metric-nlp-conf').textContent = result.nlp.confidence;
            const ns = $('#nlp-status');
            if (ns) { ns.textContent = 'Active'; ns.style.cssText = 'color:#10B981'; }
        }

        // Feature importance
        if (result.featureImportances) {
            const container = $('#feature-importance');
            container.innerHTML = result.featureImportances.map(f => `
                <div class="feature-item">
                    <span class="feature-name">${f.name}</span>
                    <div class="feature-bar">
                        <div class="feature-bar-fill" style="width:${f.importance * 100}%"></div>
                    </div>
                    <span class="feature-val">${typeof f.value === 'number' ? f.value.toFixed(2) : f.value}</span>
                </div>
            `).join('');
        }

        // Timeline chart
        drawTimeline();
    }

    function updateSessionDuration() {
        if (!state.sessionStart) return;
        const elapsed = Math.floor((Date.now() - state.sessionStart) / 1000);
        const mins = Math.floor(elapsed / 60);
        const secs = elapsed % 60;
        $('#session-duration').textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    function updateSessionDurationLoop() {
        const ticker = setInterval(() => {
            if (!state.sessionActive) { clearInterval(ticker); return; }
            updateSessionDuration();
        }, 1000);
    }

    // ─── Timeline Chart ───────────────────────────────────────
    function drawTimeline() {
        const canvas = $('#timeline-canvas');
        if (!canvas || state.results.length === 0) return;

        const ctx = canvas.getContext('2d');
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width - 48;
        canvas.height = 180;

        const w = canvas.width;
        const h = canvas.height;
        const padding = { top: 20, right: 20, bottom: 30, left: 40 };
        const chartW = w - padding.left - padding.right;
        const chartH = h - padding.top - padding.bottom;

        ctx.clearRect(0, 0, w, h);

        // Grid lines
        ctx.strokeStyle = 'rgba(255,255,255,0.05)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = padding.top + (chartH / 4) * i;
            ctx.beginPath();
            ctx.moveTo(padding.left, y);
            ctx.lineTo(w - padding.right, y);
            ctx.stroke();
        }

        // Y-axis labels
        ctx.fillStyle = 'rgba(255,255,255,0.3)';
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.textAlign = 'right';
        const labels = ['Deep', 'Soft', 'Dist', 'Switch', 'Pseudo'];
        labels.forEach((label, i) => {
            ctx.fillText(label, padding.left - 5, padding.top + (chartH / 4) * i + 4);
        });

        if (state.results.length < 2) return;

        // Draw flow state line
        const colors = ['#EF4444', '#F59E0B', '#F97316', '#10B981', '#8B5CF6'];
        const points = state.results.map((r, i) => ({
            x: padding.left + (i / (state.results.length - 1)) * chartW,
            y: padding.top + ((4 - r.flowState) / 4) * chartH,
            state: r.flowState,
        }));

        // Gradient fill under line
        const gradient = ctx.createLinearGradient(0, padding.top, 0, h - padding.bottom);
        gradient.addColorStop(0, 'rgba(139, 92, 246, 0.15)');
        gradient.addColorStop(1, 'rgba(139, 92, 246, 0)');

        ctx.beginPath();
        ctx.moveTo(points[0].x, h - padding.bottom);
        points.forEach(p => ctx.lineTo(p.x, p.y));
        ctx.lineTo(points[points.length - 1].x, h - padding.bottom);
        ctx.fillStyle = gradient;
        ctx.fill();

        // Draw line
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
            const cp1x = (points[i - 1].x + points[i].x) / 2;
            ctx.bezierCurveTo(cp1x, points[i - 1].y, cp1x, points[i].y, points[i].x, points[i].y);
        }
        ctx.strokeStyle = '#8B5CF6';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw points
        points.forEach(p => {
            ctx.beginPath();
            ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
            ctx.fillStyle = colors[p.state];
            ctx.fill();
            ctx.strokeStyle = 'rgba(255,255,255,0.3)';
            ctx.lineWidth = 1;
            ctx.stroke();
        });
    }

    // ─── Audio Visualizer ─────────────────────────────────────
    function setupAudioVisualizer() {
        const container = $('#audio-visualizer');
        if (!container) return;

        // Create frequency bars
        for (let i = 0; i < 32; i++) {
            const bar = document.createElement('div');
            bar.className = 'viz-bar';
            bar.style.cssText = `
                display: inline-block;
                width: 3px;
                height: 2px;
                margin: 0 1px;
                background: linear-gradient(to top, #06B6D4, #8B5CF6);
                border-radius: 2px;
                transition: height 0.05s ease;
                vertical-align: bottom;
            `;
            container.appendChild(bar);
        }

        // Add audio level indicator below the bars
        const levelRow = document.createElement('div');
        levelRow.id = 'audio-level-row';
        levelRow.style.cssText = `
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 8px;
            padding: 6px 0;
        `;
        levelRow.innerHTML = `
            <span id="mic-live-dot" style="
                width: 8px; height: 8px; border-radius: 50%;
                background: #6b7280; flex-shrink: 0;
                transition: background 0.2s, box-shadow 0.2s;
            "></span>
            <div id="audio-level-bar" style="
                flex: 1; height: 6px; border-radius: 3px;
                background: rgba(255,255,255,0.05); overflow: hidden;
            ">
                <div id="audio-level-fill" style="
                    width: 0%; height: 100%; border-radius: 3px;
                    background: linear-gradient(90deg, #06B6D4, #8B5CF6);
                    transition: width 0.08s ease;
                "></div>
            </div>
            <span id="mic-live-label" style="
                font-size: 0.6rem; font-weight: 700; color: #6b7280;
                font-family: 'JetBrains Mono', monospace;
                letter-spacing: 0.05em;
            ">MIC OFF</span>
        `;
        container.parentElement.appendChild(levelRow);
    }

    /** Start live audio level — runs immediately when mic is enabled, even before session */
    function startLiveAudioLevel() {
        const fill = $('#audio-level-fill');
        const dot = $('#mic-live-dot');
        const label = $('#mic-live-label');

        if (!fill || !dot || !label) return;

        // Mark mic as live
        label.textContent = 'LIVE';
        label.style.color = '#10B981';
        dot.style.background = '#10B981';
        dot.style.boxShadow = '0 0 8px rgba(16, 185, 129, 0.6)';

        function updateLevel() {
            if (!state.micEnabled) {
                fill.style.width = '0%';
                dot.style.background = '#6b7280';
                dot.style.boxShadow = 'none';
                label.textContent = 'MIC OFF';
                label.style.color = '#6b7280';
                return;
            }

            // Get real audio data
            const data = state.pipeline.getAudioFrequencyData();
            if (data.length > 0) {
                // Compute RMS level from frequency data
                let sum = 0;
                for (let i = 0; i < data.length; i++) sum += data[i];
                const avg = sum / data.length;
                const level = Math.min(100, (avg / 128) * 100);

                fill.style.width = level + '%';

                // Pulse dot when detecting sound
                if (level > 10) {
                    dot.style.background = '#10B981';
                    dot.style.boxShadow = '0 0 12px rgba(16, 185, 129, 0.8)';
                    label.textContent = 'HEARING YOU';
                    label.style.color = '#10B981';
                } else {
                    dot.style.background = '#F59E0B';
                    dot.style.boxShadow = '0 0 6px rgba(245, 158, 11, 0.4)';
                    label.textContent = 'LISTENING...';
                    label.style.color = '#F59E0B';
                }
            }

            requestAnimationFrame(updateLevel);
        }
        requestAnimationFrame(updateLevel);
    }

    function startAudioVisualizerLoop() {
        function update() {
            if (!state.sessionActive || !state.micEnabled) return;
            const data = state.pipeline.getAudioFrequencyData();
            const bars = $$('#audio-visualizer .viz-bar');
            const step = Math.floor(data.length / bars.length) || 1;
            bars.forEach((bar, i) => {
                const val = data[i * step] || 0;
                bar.style.height = Math.max(2, val / 255 * 40) + 'px';
            });
            requestAnimationFrame(update);
        }
        requestAnimationFrame(update);
    }

    // ─── Settings ─────────────────────────────────────────────
    function setupSettings() {
        $('#btn-save-profile')?.addEventListener('click', () => {
            const name = $('#settings-name')?.value || 'User';
            state.profile.userId = name;
            state.profile.save();
            updateUserBadge();
            showToast('✅', 'Profile saved');
        });

        $('#btn-clear-data')?.addEventListener('click', async () => {
            if (confirm('Clear all session data and calibration?')) {
                await state.profile.clearAll();
                showToast('🗑️', 'All data cleared');
                updateUserBadge();
            }
        });

        $('#btn-export')?.addEventListener('click', async () => {
            const csv = await state.profile.exportCSV();
            if (!csv) { showToast('⚠️', 'No data to export'); return; }
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url; a.download = 'ani_sessions.csv'; a.click();
            URL.revokeObjectURL(url);
            showToast('📥', 'CSV exported');
        });

        const nameInput = $('#settings-name');
        if (nameInput) {
            nameInput.value = state.profile.userId !== 'default' ? state.profile.userId : '';
        }
    }

    // ─── History ──────────────────────────────────────────────
    async function loadHistory() {
        const sessions = await state.profile.getAllSessions();
        const container = $('#history-list');
        if (!container) return;

        if (sessions.length === 0) {
            container.innerHTML = '<p class="empty-state">No sessions recorded yet.</p>';
            return;
        }

        const stateColors = { PSEUDO_WORKING: '#EF4444', TASK_SWITCHING: '#F59E0B', DISTRACTED: '#F97316', SOFT_FLOW: '#10B981', DEEP_FLOW: '#8B5CF6' };

        container.innerHTML = sessions.reverse().slice(0, 50).map(s => {
            const date = new Date(s.date || s.timestamp).toLocaleDateString();
            const duration = s.duration ? `${Math.floor(s.duration / 60)}m` : '—';
            const color = stateColors[s.dominantState] || '#6b7280';
            const quality = s.averageQuality !== undefined ? `${(s.averageQuality * 100).toFixed(0)}%` : '—';
            return `
                <div class="history-item">
                    <span style="color:${color};font-weight:600">${s.dominantState?.replace('_', ' ') || '—'}</span>
                    <span style="color:var(--text-muted)">${s.task?.substring(0, 60) || '—'}</span>
                    <span>${date} · ${duration}</span>
                    <span style="font-family:'JetBrains Mono',monospace;color:var(--accent-purple)">${quality}</span>
                </div>
            `;
        }).join('');
    }

    // ─── Helpers ──────────────────────────────────────────────
    function updateUserBadge() {
        const consistency = state.profile.classifyConsistency();
        const consistencyColors = {
            'IMPROVING': '#10B981', 'STABLE': '#8B5CF6',
            'DECLINING': '#EF4444', 'INCONSISTENT': '#F59E0B',
            'INSUFFICIENT_DATA': '#6b7280'
        };
        const tag = $('#consistency-tag');
        if (tag) {
            tag.textContent = consistency.replace('_', ' ');
            tag.style.color = consistencyColors[consistency] || '#6b7280';
            tag.style.background = `${consistencyColors[consistency] || '#6b7280'}20`;
        }
        const name = $('#user-name');
        if (name) name.textContent = state.profile.userId !== 'default' ? state.profile.userId : 'User';
    }

    function updateModelStatus(status, text) {
        const indicator = $('#model-status-indicator');
        if (!indicator) return;
        const dot = indicator.querySelector('.status-dot');
        const span = indicator.querySelector('span:last-child');
        if (dot) { dot.className = 'status-dot ' + status; }
        if (span) span.textContent = text;
    }

    function showToast(icon, message, duration = 3000) {
        const container = $('#toast-container');
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.innerHTML = `<span class="toast-icon">${icon}</span><span>${message}</span>`;
        container.appendChild(toast);
        setTimeout(() => {
            toast.classList.add('fade-out');
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }

    // ─── Start ────────────────────────────────────────────────
    document.addEventListener('DOMContentLoaded', init);
    if (document.readyState !== 'loading') init();
})();
