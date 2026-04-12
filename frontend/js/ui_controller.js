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
        intervalMs: 5000,
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
        await state.pipeline.init(true);

        updateLoadingStatus('Initializing Flow Guardian...', 70);
        state.guardian = new AniGuardian();

        updateLoadingStatus('Setting up interface...', 85);
        setupNavigation();
        setupSessionControls();
        setupSettings();
        setupAudioVisualizer();
        setupGuardianControls();
        startLiveDiagnosticsLoop();
        updateUserBadge();

        // Start live diagnostics feed rendering loop
        startLiveDiagnosticsLoop();

        // Update model status badges after init
        updateModelLoadStatus();

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

    // ─── Live Diagnostics Loop ────────────────────────────────
    let diagnosticsLoopRunning = false;
    function startLiveDiagnosticsLoop() {
        if (diagnosticsLoopRunning) return;
        diagnosticsLoopRunning = true;
        
        const canvas = $('#yolo-live-feed');
        if (!canvas) return;

        let lastTime = performance.now();
        let frames = 0;
        let fps = 0;

        function render() {
            // Only draw if the diagnostics panel is active
            if (state.activePanel === 'diagnostics') {
                const vp = state.pipeline?.getVisionPreprocessor();
                if (vp && vp.isActive) {
                    vp.drawDetections(canvas);
                    
                    // Box mapping is done inside drawDetections, we just calculate FPS here
                    frames++;
                    const now = performance.now();
                    if (now - lastTime >= 1000) {
                        fps = Math.round((frames * 1000) / (now - lastTime));
                        frames = 0;
                        lastTime = now;
                        const fpsLabel = $('#live-framerate');
                        if (fpsLabel) fpsLabel.textContent = `${fps} FPS`;
                    }
                } else {
                    // Blank canvas text if model not active
                    const ctx = canvas.getContext('2d');
                    ctx.fillStyle = '#1e1e2d';
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    ctx.fillStyle = '#6b7280';
                    ctx.font = '16px "JetBrains Mono"';
                    ctx.textAlign = 'center';
                    ctx.fillText('Waiting for Webcam/Model init...', canvas.width / 2, canvas.height / 2);
                    
                    // Auto-request webcam if we switch to this panel and it's not active
                    const vp = state.pipeline?.getVisionPreprocessor();
                    if (!state.webcamEnabled && state.pipeline && !state.pipeline.demoMode && !window.isWebcamRequesting) {
                        window.isWebcamRequesting = true;
                        state.pipeline.enableWebcam().then(ok => {
                            if (ok) {
                                state.webcamEnabled = true;
                                const btn = $('#btn-webcam');
                                if (btn) {
                                    btn.innerHTML = '<span>📷</span> Webcam Active';
                                    btn.classList.add('btn-primary');
                                    btn.classList.remove('btn-outline');
                                }
                            }
                            // Don't reset isWebcamRequesting right away. Once it's prompted, leave it true or reset on fail.
                            if (!ok) window.isWebcamRequesting = false;
                        });
                    }
                }
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
            
            // ── DIP DETECTION: Compare against previous result ──
            const prev = state.results.length > 0 ? state.results[state.results.length - 1] : null;
            if (prev && result.flowState < prev.flowState) {
                // Flow dropped — figure out exactly WHY from sensor data
                const reason = buildDipReason(result);
                if (reason) {
                    showDipNotification(reason, prev.flowState, result.flowState);
                }
            }
            
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

    /**
     * Inspect the raw sensor data and return a human-readable string 
     * explaining what single trigger caused the graph to dip.
     */
    function buildDipReason(result) {
        // Priority order: most critical trigger first
        if (result.vision?.phoneVisible === 'Yes') {
            return { icon: '📱', trigger: 'Phone detected on your desk', advice: 'Put it away — every glance costs ~23 min of focus.' };
        }
        if (result.vision?.activeTabCategory === 'distraction') {
            const tabName = result.vision.activeTabTitle || 'a distraction site';
            // Extract site name from "Video Title - YouTube" style strings
            const site = tabName.split(/[-–—|]/).pop().trim();
            return { icon: '🌐', trigger: `You switched to ${site}`, advice: 'Close it and get back to your task.' };
        }
        if (result.screen?.className === 'DISTRACTION') {
            return { icon: '🎮', trigger: 'Screen shows non-productive content', advice: 'Switch back to your work window.' };
        }
        if (result.vision?.tabCount > 20) {
            return { icon: '📑', trigger: `${result.vision.tabCount} tabs open`, advice: 'Close tabs you\'re not using — tab overload kills focus.' };
        }
        if (result.vision?.switchRate > 5) {
            return { icon: '🔄', trigger: `Rapid tab switching detected (${result.vision.switchRate}/min)`, advice: 'Pick one task and stay on it.' };
        }
        if (result.audio?.energyLevel === 'Silent' && result.audio?.activity === '0%') {
            return { icon: '😴', trigger: 'No activity detected — you may have zoned out', advice: 'Take a stretch, then re-engage.' };
        }
        // Generic fallback
        return { icon: '📉', trigger: 'Focus quality dropped', advice: 'Check your environment for distractions.' };
    }

    /** 
     * Show a prominent, styled notification when the graph dips.
     * Different from a regular toast — it's bigger, red-tinted, and stays longer.
     */
    function showDipNotification(reason, fromState, toState) {
        const stateNames = ['Pseudo-Working', 'Task-Switching', 'Distracted', 'Soft Flow', 'Deep Flow'];
        const container = $('#toast-container');
        
        const toast = document.createElement('div');
        toast.className = 'toast dip-toast';
        toast.innerHTML = `
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
                <span style="font-size:1.6rem;">${reason.icon}</span>
                <div>
                    <div style="font-weight:700; color:#FCA5A5; font-size:0.95rem;">⚠ Flow Dip Detected</div>
                    <div style="font-size:0.7rem; color:#9ca3af;">${stateNames[fromState]} → ${stateNames[toState]}</div>
                </div>
            </div>
            <div style="font-size:0.85rem; color:#e5e7eb; margin-bottom:4px;">${reason.trigger}</div>
            <div style="font-size:0.75rem; color:#fbbf24; font-style:italic;">${reason.advice}</div>
        `;
        toast.style.cssText = `
            background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(30,30,45,0.95));
            border: 1px solid rgba(239,68,68,0.4);
            border-left: 4px solid #EF4444;
            padding: 14px 18px;
            border-radius: 12px;
            backdrop-filter: blur(12px);
            animation: slideInRight 0.4s ease;
            max-width: 380px;
            box-shadow: 0 8px 32px rgba(239,68,68,0.2);
        `;
        container.appendChild(toast);

        // Stay visible for 8 seconds (longer than normal toasts)
        setTimeout(() => {
            toast.classList.add('fade-out');
            setTimeout(() => toast.remove(), 400);
        }, 8000);
    }

    // ─── Guardian Display ─────────────────────────────────────
    function displayGuardianResponse(response) {
        // Add message
        addGuardianMessage(response.emoji, response.message, response.mood, response.evidence);

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

    function addGuardianMessage(emoji, text, mood = '', evidence = '') {
        const container = $('#guardian-messages');
        if (!container) return;

        const msg = document.createElement('div');
        msg.className = `guardian-message${mood ? ` mood-${mood}` : ''}`;
        
        let evidenceHtml = '';
        if (evidence) {
            evidenceHtml = `<div class="guardian-evidence-box" style="margin-top: 8px; font-size: 0.75rem; color: #a0a0b0; background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px; border-left: 2px solid rgba(255,255,255,0.1);">
                ${evidence}
            </div>`;
        }

        msg.innerHTML = `
            <div style="display:flex;">
                <span class="guardian-msg-emoji">${emoji}</span>
                <span class="guardian-msg-text">${text}</span>
            </div>
            ${evidenceHtml}
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

        // ─── VISION CARD ─────────────────────────────────────
        if (result.vision) {
            $('#metric-tabs').textContent = result.vision.tabCount;
            $('#metric-phone').textContent = result.vision.phoneVisible;
            $('#metric-distractions').textContent = result.vision.distractions;
            $('#metric-focus').textContent = result.vision.focusRatio;

            // Phone highlight color
            const phoneEl = $('#metric-phone');
            if (phoneEl) {
                if (result.vision.phoneVisible === 'Yes') {
                    phoneEl.style.color = '#EF4444';
                    phoneEl.textContent = '📱 Yes';
                } else {
                    phoneEl.style.color = '#10B981';
                    phoneEl.textContent = '✅ No';
                }
            }

            // Source badge
            updateSourceBadge('vision-source-badge', result.vision.source);

            // YOLO detection list + Extension tab analysis
            const detList = $('#yolo-detection-list');
            if (detList) {
                let html = '';

                // Extension tab categories
                if (result.vision.extensionConnected && result.vision.tabCategories) {
                    const cats = result.vision.tabCategories;
                    const catEmojis = { productive: '✅', distraction: '⚠️', communication: '💬', news: '📰', neutral: '📄' };
                    const catColors = { productive: '#10B981', distraction: '#EF4444', communication: '#06B6D4', news: '#F59E0B', neutral: '#6b7280' };
                    const total = result.vision.tabCount || 1;

                    html += '<span class="detail-label" style="margin-top:0;">Tab Categories:</span>';
                    for (const [cat, count] of Object.entries(cats)) {
                        if (count === 0) continue;
                        const pct = ((count / total) * 100).toFixed(0);
                        html += `<div class="det-item">
                            <span class="det-icon">${catEmojis[cat] || '📄'}</span>
                            <span class="det-name">${cat}</span>
                            <span class="det-conf" style="color:${catColors[cat]}">${count}</span>
                            <div class="det-conf-bar"><div class="det-conf-fill" style="width:${pct}%;background:${catColors[cat]}"></div></div>
                        </div>`;
                    }

                    // Active tab
                    if (result.vision.activeTabTitle) {
                        html += `<div style="margin-top:6px;padding:5px 8px;background:rgba(139,92,246,0.06);border-radius:6px;border:1px solid rgba(139,92,246,0.1);">
                            <span style="font-size:0.6rem;color:#8B5CF6;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;">▶ Active:</span>
                            <span style="font-size:0.7rem;color:#e0e0e8;margin-left:4px;">${result.vision.activeTabTitle.substring(0, 50)}</span>
                        </div>`;
                    }

                    // Switch rate
                    if (result.vision.switchRate > 0) {
                        html += `<div style="font-size:0.65rem;color:#6b7280;margin-top:4px;">Tab switches/min: <span style="color:#F59E0B;font-weight:700;">${result.vision.switchRate}</span></div>`;
                    }
                }

                // YOLO detections (from webcam)
                const dets = result.vision.detections || [];
                if (dets.length > 0) {
                    html += '<span class="detail-label" style="margin-top:8px;">YOLO Detections:</span>';
                    const detEmojis = { phone: '📱', monitor: '🖥️', work_tool: '⌨️', distraction: '⚠️' };
                    html += dets.map(d =>
                        `<div class="det-item">
                            <span class="det-icon">${detEmojis[d.className] || '🔍'}</span>
                            <span class="det-name">${d.className}</span>
                            <span class="det-conf">${(d.confidence * 100).toFixed(0)}%</span>
                            <div class="det-conf-bar"><div class="det-conf-fill" style="width:${d.confidence * 100}%"></div></div>
                        </div>`
                    ).join('');
                } else if (!result.vision.extensionConnected) {
                    html += '<span class="empty-det">Install Chrome extension for tab analysis</span>';
                }

                detList.innerHTML = html;
            }
            
            // ─── EXTENSION TELEMETRY MODULE ────────────────────
            const extStatus = $('#ext-status-badge');
            if (extStatus) {
                if (result.vision.extensionConnected) {
                    extStatus.textContent = '🟢 Connected & Streaming';
                    extStatus.className = 'badge badge-primary';
                    $('#ext-tab-count').textContent = result.vision.tabCount || 0;
                    $('#ext-prod-score').textContent = result.vision.productivityScore !== null ? Math.round(result.vision.productivityScore * 100) + '%' : '-';
                    $('#ext-dist-tabs').textContent = (result.vision.tabCategories?.distraction || 0) + (result.vision.tabCategories?.news || 0);
                    $('#ext-comm-tabs').textContent = result.vision.tabCategories?.communication || 0;
                    $('#ext-active-url').textContent = result.vision.activeTabUrl || result.vision.activeTabTitle || 'Unknown';
                    $('#ext-active-url').title = result.vision.activeTabUrl || result.vision.activeTabTitle || '';
                } else {
                    extStatus.textContent = '🔴 Offline (Waiting for extension...)';
                    extStatus.className = 'badge badge-danger';
                }
            }
        }

        // ─── SCREEN CARD ─────────────────────────────────────
        if (result.screen) {
            const screenClassLabels = {
                'PRODUCTIVE_CODE': '🖥️ Coding',
                'PRODUCTIVE_DOCS': '📄 Documents',
                'COMMUNICATION': '💬 Communication',
                'DISTRACTION': '🎮 Distraction',
                'NEUTRAL': '📂 Neutral',
            };
            const screenClassColors = {
                'PRODUCTIVE_CODE': '#10B981',
                'PRODUCTIVE_DOCS': '#3B82F6',
                'COMMUNICATION': '#F59E0B',
                'DISTRACTION': '#EF4444',
                'NEUTRAL': '#6B7280',
            };

            const screenLabel = screenClassLabels[result.screen.className] || result.screen.className;
            const screenColor = screenClassColors[result.screen.className] || '#8B5CF6';

            $('#metric-screen-class').textContent = screenLabel;
            $('#metric-screen-class').style.color = screenColor;
            $('#metric-screen-prod').textContent = result.screen.productivityScore;
            $('#metric-screen-conf').textContent = result.screen.confidence;

            // Source badge
            updateSourceBadge('screen-source-badge', result.screen.source);

            // Probability bars
            const screenProbBars = $('#screen-prob-bars');
            if (screenProbBars && result.screen.rawProbs) {
                const classNames = ['Code', 'Docs', 'Chat', 'Distract', 'Neutral'];
                const classColors = ['#10B981', '#3B82F6', '#F59E0B', '#EF4444', '#6B7280'];
                let html = '';
                for (let i = 0; i < classNames.length; i++) {
                    const pct = (result.screen.rawProbs[i] * 100).toFixed(0);
                    html += `<div class="prob-row" style="margin-bottom:3px;">
                        <span style="font-size:0.65rem;color:#9ca3af;width:50px;display:inline-block;">${classNames[i]}</span>
                        <div class="prob-bar" style="flex:1;height:8px;background:rgba(255,255,255,0.05);border-radius:4px;margin:0 6px;">
                            <div style="width:${pct}%;height:100%;background:${classColors[i]};border-radius:4px;transition:width 0.4s ease;"></div>
                        </div>
                        <span style="font-size:0.6rem;color:#9ca3af;width:28px;text-align:right;">${pct}%</span>
                    </div>`;
                }
                screenProbBars.innerHTML = html;
            }
        } else {
            // No screen capture active
            const screenClass = $('#metric-screen-class');
            if (screenClass) {
                screenClass.textContent = 'No Screen Share';
                screenClass.style.color = '#6B7280';
            }
            const screenProd = $('#metric-screen-prod');
            if (screenProd) screenProd.textContent = '—';
            const screenConf = $('#metric-screen-conf');
            if (screenConf) screenConf.textContent = '—';
            updateSourceBadge('screen-source-badge', 'idle');
        }

        // ─── AUDIO CARD ──────────────────────────────────────
        if (result.audio) {
            $('#metric-speech').textContent = result.audio.energyLevel || result.audio.speechClass;
            $('#metric-wpm').textContent = result.audio.tone || '—';
            $('#metric-fluency').textContent = result.audio.activity || '—';
            $('#metric-audio-conf').textContent = result.audio.confidence;

            // Source badge
            updateSourceBadge('audio-source-badge', result.audio.source);

            // Voice state visualization bars
            const audioProbContainer = $('#audio-prob-bars');
            if (audioProbContainer && result.audio.energyLevel) {
                const energyLevels = ['Silent', 'Quiet', 'Active', 'Energized'];
                const toneLabels = ['Calm', 'Neutral', 'Animated', 'Stressed'];
                const currentEnergy = energyLevels.indexOf(result.audio.energyLevel);
                const currentTone = toneLabels.indexOf(result.audio.tone);
                
                // Energy bar
                const energyPercent = Math.max(5, ((currentEnergy + 1) / 4) * 100);
                // Activity bar 
                const activityVal = parseInt(result.audio.activity) || 0;
                
                audioProbContainer.innerHTML = `
                    <div class="class-prob-row ${currentEnergy >= 0 ? 'active' : ''}">
                        <span class="cp-name">Energy</span>
                        <div class="cp-bar"><div class="cp-fill" style="width:${energyPercent}%"></div></div>
                        <span class="cp-val">${result.audio.energyLevel}</span>
                    </div>
                    <div class="class-prob-row ${currentTone >= 0 ? 'active' : ''}">
                        <span class="cp-name">Tone</span>
                        <div class="cp-bar"><div class="cp-fill" style="width:${Math.max(5, ((currentTone + 1) / 4) * 100)}%"></div></div>
                        <span class="cp-val">${result.audio.tone}</span>
                    </div>
                    <div class="class-prob-row active">
                        <span class="cp-name">Activity</span>
                        <div class="cp-bar"><div class="cp-fill" style="width:${Math.max(5, activityVal)}%"></div></div>
                        <span class="cp-val">${activityVal}%</span>
                    </div>
                `;
            } else if (audioProbContainer) {
                audioProbContainer.innerHTML = '<span class="empty-det">Enable mic to see voice state</span>';
            }
        }

        // ─── NLP CARD ────────────────────────────────────────
        if (result.nlp) {
            $('#metric-task-type').textContent = result.nlp.taskType;
            $('#metric-demand').textContent = result.nlp.demand;
            $('#metric-nlp-conf').textContent = result.nlp.confidence;

            // Source badge
            updateSourceBadge('nlp-source-badge', result.nlp.source);

            // Class probability bars
            const nlpProbs = result.nlp.classProbs;
            const nlpProbContainer = $('#nlp-prob-bars');
            if (nlpProbContainer && nlpProbs) {
                const nlpClasses = ['Deep Work', 'Shallow', 'Creative', 'Admin', 'Comms'];
                nlpProbContainer.innerHTML = nlpProbs.map((p, i) => `
                    <div class="class-prob-row ${i === result.nlp.features.task_class ? 'active' : ''}">
                        <span class="cp-name">${nlpClasses[i]}</span>
                        <div class="cp-bar"><div class="cp-fill" style="width:${p * 100}%"></div></div>
                        <span class="cp-val">${(p * 100).toFixed(0)}%</span>
                    </div>
                `).join('');
            } else if (nlpProbContainer) {
                nlpProbContainer.innerHTML = '<span class="empty-det">No NLP probabilities</span>';
            }
        }

        // ─── META CARD ───────────────────────────────────────
        if (result.meta) {
            $('#metric-meta-flow').textContent = result.meta.flowState || result.flowLabel;
            $('#metric-meta-conf').textContent = `${(result.confidence * 100).toFixed(0)}%`;

            // Source badge
            updateSourceBadge('meta-source-badge', result.meta.source);

            // Flow probability bars
            const metaProbContainer = $('#meta-prob-bars');
            if (metaProbContainer && result.meta.classProbs) {
                const flowNames = ['Pseudo', 'Switch', 'Distracted', 'Soft Flow', 'Deep Flow'];
                const flowEmojis = ['🔴', '🟠', '🟡', '🟢', '🟣'];
                metaProbContainer.innerHTML = result.meta.classProbs.map((p, i) => `
                    <div class="class-prob-row ${i === result.flowState ? 'active' : ''}">
                        <span class="cp-name">${flowEmojis[i]} ${flowNames[i]}</span>
                        <div class="cp-bar"><div class="cp-fill" style="width:${p * 100}%"></div></div>
                        <span class="cp-val">${(p * 100).toFixed(0)}%</span>
                    </div>
                `).join('');
            }
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

    /** Update a source badge element */
    function updateSourceBadge(id, source) {
        const el = $(`#${id}`);
        if (!el) return;
        
        if (source === 'ONNX Model') {
            el.textContent = 'ONNX Model ✅';
            el.className = 'modality-source-badge badge-onnx';
        } else if (source === 'Chrome Extension') {
            el.textContent = 'Extension ✅';
            el.className = 'modality-source-badge badge-onnx';
        } else if (source && source.includes('+ Extension')) {
            el.textContent = 'ONNX + Ext ✅';
            el.className = 'modality-source-badge badge-onnx';
        } else if (source === 'no-webcam' || source === 'no-mic' || source === 'no-text') {
            const missing = source === 'no-webcam' ? 'Enable Webcam' : source === 'no-mic' ? 'Enable Mic' : 'Enter Task';
            el.textContent = missing;
            el.className = 'modality-source-badge badge-warning';
        } else if (source === 'no-model') {
            el.textContent = 'No Model';
            el.className = 'modality-source-badge badge-error';
        } else if (source === 'keyword-demo') {
            el.textContent = 'Keyword Demo';
            el.className = 'modality-source-badge badge-demo';
        } else if (source === 'error') {
            el.textContent = 'Error ❌';
            el.className = 'modality-source-badge badge-error';
        } else {
            el.textContent = source || 'Idle';
            el.className = 'modality-source-badge badge-idle';
        }
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

        // Draw dip annotations — red dashed lines + reason icons at each drop
        for (let i = 1; i < state.results.length; i++) {
            const prev = state.results[i - 1];
            const curr = state.results[i];
            if (curr.flowState < prev.flowState) {
                const p = points[i];
                // Vertical dashed red line
                ctx.save();
                ctx.setLineDash([4, 4]);
                ctx.strokeStyle = 'rgba(239, 68, 68, 0.5)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(p.x, padding.top);
                ctx.lineTo(p.x, h - padding.bottom);
                ctx.stroke();
                ctx.restore();

                // Reason icon at the top of the dashed line
                const reason = buildDipReason(curr);
                ctx.font = '14px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText(reason.icon, p.x, padding.top - 4);
            }
        }
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

    /** Update UI to reflect which models actually loaded */
    function updateModelLoadStatus() {
        if (!state.pipeline) return;
        const models = state.pipeline.models || {};
        const loaded = Object.keys(models);
        const total = 3; // Vision, Audio, Meta (NLP moved to keyword engine)

        // Update sidebar status indicator
        if (loaded.length >= 2) {
            updateModelStatus('online', `${loaded.length}/${total} models loaded`);
        } else {
            updateModelStatus('offline', 'Models offline');
        }

        // Update demo mode checkbox to match actual state
        const demoCheckbox = $('#settings-demo');
        if (demoCheckbox) {
            demoCheckbox.checked = state.pipeline.demoMode;
        }

        // Update individual model status badges in the Models panel
        const statusMap = {
            vision: '#vision-model-status',
            audio: '#audio-model-status',
            meta: '#meta-model-status',
        };

        for (const [modelName, selector] of Object.entries(statusMap)) {
            const row = $(selector);
            if (!row) continue;
            const tag = row.querySelector('.tag');
            if (!tag) continue;

            if (models[modelName]) {
                tag.textContent = 'Loaded ✅';
                tag.className = 'tag tag-online';
                tag.style.cssText = 'background: rgba(16,185,129,0.15); color: #10B981; border: 1px solid rgba(16,185,129,0.3);';
            } else {
                tag.textContent = 'Not loaded';
                tag.className = 'tag tag-offline';
                tag.style.cssText = '';
            }
        }
        
        // NLP is now a zero-dependency Keyword Engine, always "loaded"
        const nlpRow = $('#nlp-model-status');
        if (nlpRow) {
            const nlpTag = nlpRow.querySelector('.tag');
            if (nlpTag) {
                nlpTag.textContent = 'Ready (Native) ✅';
                nlpTag.className = 'tag tag-online';
                nlpTag.style.cssText = 'background: rgba(16,185,129,0.15); color: #10B981; border: 1px solid rgba(16,185,129,0.3);';
            }
        }
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
