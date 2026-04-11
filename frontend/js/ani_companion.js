/**
 * AniCompanion — Live2D Hiyori character integration with voice + PiP.
 * 
 * Renders the Hiyori Live2D model inside the Flow State card,
 * speaks guardian messages via high-quality browser TTS,
 * and supports Picture-in-Picture mode for persistent visibility.
 * 
 * Dependencies (loaded via CDN in index.html):
 *   - PixiJS v7
 *   - live2dcubismcore.min.js (Cubism 4 Core)
 *   - pixi-live2d-display/dist/cubism4.min.js
 */
class AniCompanion {
    constructor() {
        this.model = null;
        this.app = null;
        this.canvas = null;
        this.pipWindow = null;
        this.isPipActive = false;
        this.isSpeaking = false;
        this.voiceReady = false;
        this.preferredVoice = null;
        this.currentMood = 'idle'; // idle, happy, concerned, alert
        this._lipSyncInterval = null;
        
        // Model path (served from project root by serve.py)
        this.modelPath = '/hiyori_free_en/runtime/hiyori_free_t08.model3.json';
        
        // Init voices
        this._initVoices();
    }

    // ─── Voice Setup ─────────────────────────────────────────

    _initVoices() {
        const pickBest = () => {
            const voices = speechSynthesis.getVoices();
            if (!voices.length) return;

            // Priority order for realistic female English voices on Windows
            const preferred = [
                'Microsoft Aria Online',     // Neural, very natural
                'Microsoft Jenny Online',    // Neural
                'Google UK English Female',  // High quality on Chrome
                'Microsoft Zira',            // Decent fallback
                'Samantha',                  // macOS
                'Google US English',         // Chrome default
            ];

            for (const name of preferred) {
                const v = voices.find(voice => voice.name.includes(name));
                if (v) {
                    this.preferredVoice = v;
                    this.voiceReady = true;
                    console.log(`[Companion] Voice selected: ${v.name} (${v.lang})`);
                    return;
                }
            }

            // Fallback: pick first English female-sounding voice
            const english = voices.filter(v => v.lang.startsWith('en'));
            if (english.length > 0) {
                this.preferredVoice = english[0];
                this.voiceReady = true;
                console.log(`[Companion] Voice fallback: ${english[0].name}`);
            }
        };

        // Voices load async on most browsers
        if (speechSynthesis.getVoices().length > 0) {
            pickBest();
        }
        speechSynthesis.onvoiceschanged = pickBest;
    }

    // ─── Live2D Model Init ───────────────────────────────────

    async init(containerId = 'companion-canvas-container') {
        const container = document.getElementById(containerId);
        if (!container) {
            console.warn('[Companion] Container not found:', containerId);
            return false;
        }

        // Check if Live2D libraries are loaded
        if (typeof PIXI === 'undefined' || !PIXI.live2d) {
            console.warn('[Companion] PixiJS or pixi-live2d-display not loaded. Companion disabled.');
            container.innerHTML = '<p style="color:#a0a0b0;text-align:center;font-size:0.8rem;padding:20px;">Live2D libraries not loaded</p>';
            return false;
        }

        try {
            // Create Pixi app inside the container
            this.canvas = document.createElement('canvas');
            this.canvas.id = 'companion-canvas';
            container.appendChild(this.canvas);

            this.app = new PIXI.Application({
                view: this.canvas,
                transparent: true,
                backgroundAlpha: 0,
                resizeTo: container,
                antialias: true,
            });

            // Load Hiyori model
            console.log('[Companion] Loading Hiyori model...');
            this.model = await PIXI.live2d.Live2DModel.from(this.modelPath);
            this.app.stage.addChild(this.model);

            // Scale and position model to fit the container
            this._fitModel();
            
            // Enable eye tracking (follows cursor)
            this.model.on('pointermove', (e) => {
                // Not needed for tracking, we'll use a different approach
            });

            // Set up cursor tracking for the model
            this._setupCursorTracking();

            // Set idle motion
            this.model.motion('Idle');

            // Observe container resizes
            if (window.ResizeObserver) {
                new ResizeObserver(() => this._fitModel()).observe(container);
            }

            // Remove the loading overlay div
            const loader = container.querySelector('.companion-loading');
            if (loader) loader.style.display = 'none';

            console.log('[Companion] ✅ Hiyori loaded successfully!');
            return true;
        } catch (e) {
            console.error('[Companion] Failed to load Live2D model:', e);
            container.innerHTML = '<p style="color:#f87171;text-align:center;font-size:0.8rem;padding:20px;">Failed to load companion model</p>';
            return false;
        }
    }

    _fitModel() {
        if (!this.model || !this.canvas) return;
        const w = this.canvas.width;
        const h = this.canvas.height;

        // Scale model to fit height with some padding
        const scale = h / this.model.height * 0.75;
        this.model.scale.set(scale);

        // Center horizontally, anchor near bottom
        this.model.x = w / 2;
        this.model.y = h * 0.9;
        this.model.anchor.set(0.5, 0.9);
    }

    _setupCursorTracking() {
        if (!this.model) return;
        
        document.addEventListener('mousemove', (e) => {
            if (!this.model) return;
            const rect = this.canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width * 2 - 1;   // -1 to 1
            const y = (e.clientY - rect.top) / rect.height * 2 - 1;
            this.model.focus(x, y);
        });
    }

    // ─── Mood / Expression System ────────────────────────────

    setMood(mood) {
        if (!this.model) return;
        this.currentMood = mood;

        switch (mood) {
            case 'happy':
            case 'pleased':
                this.model.motion('Idle', 0);
                break;
            case 'concerned':
            case 'worried':
                this.model.motion('Flick', 0);       // Surprised / alert motion
                break;
            case 'alert':
                this.model.motion('FlickDown', 0);    // Stern look
                break;
            case 'tap':
                this.model.motion('Tap', 0);
                break;
            default:
                this.model.motion('Idle');
                break;
        }
    }

    // ─── Voice / Speech System (Web Speech API) ──────────────

    /**
     * Speak a message using the best available voice.
     * Uses pitch/rate tuning to sound more like an anime companion.
     * @param {string} text - The message to speak
     * @param {string} emotion - 'calm', 'urgent', 'happy', 'supportive'
     * @returns {Promise<void>}
     */
    speak(text, emotion = 'calm') {
        return new Promise((resolve) => {
            if (!this.voiceReady || !text) {
                resolve();
                return;
            }

            // Cancel any ongoing speech
            speechSynthesis.cancel();

            // Clean text of emoji and markdown
            const cleanText = text
                .replace(/[\u{1F000}-\u{1FFFF}]/gu, '')
                .replace(/\*\*/g, '')
                .replace(/📉|🔥|🟣|🟢|💜|🧠|📱|🚫|⚠️|🌊|🧘|🚨|🔄|🎯|💪|🧩|🤔|💡|👋|🚀|✨/g, '')
                .replace(/\s+/g, ' ')
                .trim();

            if (!cleanText) {
                resolve();
                return;
            }

            const utterance = new SpeechSynthesisUtterance(cleanText);
            utterance.voice = this.preferredVoice;
            utterance.lang = 'en-US';

            // Emotion-based tuning for anime-like quality
            switch (emotion) {
                case 'urgent':
                    utterance.rate = 1.15;
                    utterance.pitch = 1.3;
                    utterance.volume = 1.0;
                    break;
                case 'happy':
                    utterance.rate = 1.05;
                    utterance.pitch = 1.25;
                    utterance.volume = 0.9;
                    break;
                case 'supportive':
                    utterance.rate = 0.95;
                    utterance.pitch = 1.15;
                    utterance.volume = 0.85;
                    break;
                case 'calm':
                default:
                    utterance.rate = 1.0;
                    utterance.pitch = 1.2;
                    utterance.volume = 0.85;
                    break;
            }

            // Lip sync: animate mouth while speaking
            utterance.onstart = () => {
                this.isSpeaking = true;
                this._startLipSync();
            };

            utterance.onend = () => {
                this.isSpeaking = false;
                this._stopLipSync();
                resolve();
            };

            utterance.onerror = () => {
                this.isSpeaking = false;
                this._stopLipSync();
                resolve();
            };

            speechSynthesis.speak(utterance);
        });
    }

    _startLipSync() {
        if (this._lipSyncInterval || !this.model) return;
        
        // Animate mouth open/close for lip sync effect
        this._lipSyncInterval = setInterval(() => {
            if (!this.model || !this.isSpeaking) {
                this._stopLipSync();
                return;
            }
            // Randomized mouth movement for natural-looking lip sync
            const openAmount = Math.random() * 0.7 + 0.3;
            try {
                this.model.internalModel?.coreModel?.setParameterValueById?.('ParamMouthOpenY', openAmount);
            } catch (e) { /* ignore if param not found */ }
        }, 80);
    }

    _stopLipSync() {
        if (this._lipSyncInterval) {
            clearInterval(this._lipSyncInterval);
            this._lipSyncInterval = null;
        }
        // Close mouth
        try {
            this.model?.internalModel?.coreModel?.setParameterValueById?.('ParamMouthOpenY', 0);
        } catch (e) { /* ignore */ }
    }

    // ─── Guardian Integration ────────────────────────────────

    /**
     * Called by the UI controller when a guardian response is generated.
     * Sets the mood, plays the motion, and speaks the message.
     */
    async onGuardianMessage(response) {
        if (!this.model) return;

        // Determine emotion from severity/mood
        let emotion = 'calm';
        let motionMood = response.mood || 'idle';

        if (response.severity === 'high') {
            emotion = 'urgent';
            motionMood = 'alert';
        } else if (response.mood === 'happy' || response.mood === 'pleased') {
            emotion = 'happy';
        } else if (response.mood === 'supportive') {
            emotion = 'supportive';
        } else if (response.mood === 'concerned' || response.mood === 'worried') {
            emotion = 'urgent';
            motionMood = 'concerned';
        }

        // Play motion
        this.setMood(motionMood);

        // Speak the message
        await this.speak(response.message, emotion);

        // Return to idle after speaking
        setTimeout(() => this.setMood('idle'), 1000);
    }

    // ─── Picture-in-Picture Mode ─────────────────────────────

    async togglePiP() {
        if (this.isPipActive) {
            await this.exitPiP();
        } else {
            await this.enterPiP();
        }
    }

    async enterPiP() {
        if (!this.canvas) return;
        
        try {
            // Method 1: Use Document PiP API (Chrome 116+)
            if ('documentPictureInPicture' in window) {
                const pipWin = await documentPictureInPicture.requestWindow({
                    width: 300,
                    height: 400,
                });

                // Style the PiP window
                const style = pipWin.document.createElement('style');
                style.textContent = `
                    body {
                        margin: 0;
                        background: #0a0a12;
                        overflow: hidden;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        height: 100vh;
                        font-family: 'Inter', sans-serif;
                    }
                    #pip-status {
                        position: absolute;
                        bottom: 8px;
                        left: 0;
                        right: 0;
                        text-align: center;
                        color: #a5b4fc;
                        font-size: 11px;
                        font-weight: 600;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                        opacity: 0.7;
                    }
                `;
                pipWin.document.head.appendChild(style);

                // Move our canvas into the PiP window
                const pipCanvas = this.canvas;
                pipCanvas.style.width = '100%';
                pipCanvas.style.height = '100%';
                pipWin.document.body.appendChild(pipCanvas);

                // Add a status bar
                const statusDiv = pipWin.document.createElement('div');
                statusDiv.id = 'pip-status';
                statusDiv.textContent = 'ANI COMPANION';
                pipWin.document.body.appendChild(statusDiv);
                this._pipStatusEl = statusDiv;

                this.pipWindow = pipWin;
                this.isPipActive = true;

                // Resize the pixi app
                setTimeout(() => {
                    this.app.renderer.resize(pipCanvas.clientWidth, pipCanvas.clientHeight);
                    this._fitModel();
                }, 100);

                // When PiP is closed, return canvas
                pipWin.addEventListener('pagehide', () => {
                    this._returnCanvasFromPip();
                });

                console.log('[Companion] Entered Document PiP mode');
                return;
            }

            // Method 2: Fallback — use video PiP with canvas stream
            if (this.canvas.captureStream) {
                const stream = this.canvas.captureStream(30);
                const video = document.createElement('video');
                video.srcObject = stream;
                video.muted = true;
                video.autoplay = true;
                video.style.display = 'none';
                document.body.appendChild(video);
                
                await video.play();
                await video.requestPictureInPicture();
                
                this._pipVideo = video;
                this.isPipActive = true;

                video.addEventListener('leavepictureinpicture', () => {
                    video.remove();
                    this._pipVideo = null;
                    this.isPipActive = false;
                });

                console.log('[Companion] Entered Video PiP mode (fallback)');
                return;
            }

            console.warn('[Companion] PiP not supported in this browser');
        } catch (e) {
            console.error('[Companion] PiP failed:', e);
        }
    }

    _returnCanvasFromPip() {
        if (!this.canvas) return;
        
        const container = document.getElementById('companion-canvas-container');
        if (container && !container.contains(this.canvas)) {
            this.canvas.style.width = '';
            this.canvas.style.height = '';
            container.appendChild(this.canvas);

            setTimeout(() => {
                this.app.renderer.resize(container.clientWidth, container.clientHeight);
                this._fitModel();
            }, 100);
        }

        this.pipWindow = null;
        this.isPipActive = false;
        this._pipStatusEl = null;
        console.log('[Companion] Returned from PiP');
    }

    async exitPiP() {
        if (this.pipWindow) {
            this.pipWindow.close();
            this._returnCanvasFromPip();
        }
        if (this._pipVideo) {
            if (document.pictureInPictureElement) {
                await document.exitPictureInPicture();
            }
            this._pipVideo.remove();
            this._pipVideo = null;
        }
        this.isPipActive = false;
    }

    /**
     * Update the PiP status text (e.g., current flow state)
     */
    updatePipStatus(text) {
        if (this._pipStatusEl) {
            this._pipStatusEl.textContent = text;
        }
    }

    // ─── Cleanup ─────────────────────────────────────────────

    destroy() {
        speechSynthesis.cancel();
        this._stopLipSync();
        this.exitPiP();
        if (this.app) {
            this.app.destroy(true);
        }
    }
}
