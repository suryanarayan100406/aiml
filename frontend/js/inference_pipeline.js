/**
 * InferencePipeline — ONNX Runtime Web orchestrator for all 5 models.
 * Runs: Vision (YOLO) + Screen (MobileNet) → Audio → NLP → Meta-classifier fusion.
 * Falls back to demo mode when ONNX models are not available.
 */
class InferencePipeline {
    constructor() {
        this.models = {};
        this.modelsLoaded = false;
        this.demoMode = true;
        this.audioExtractor = null;
        this.visionPreprocessor = null;
        this.nlpTokenizer = null;
        this.screenClassifier = null;  // Screen productivity classifier
        this.flowStateNames = ['PSEUDO_WORKING', 'TASK_SWITCHING', 'DISTRACTED', 'SOFT_FLOW', 'DEEP_FLOW'];
        this.flowEmojis = ['🔴', '🟠', '🟡', '🟢', '🟣'];
        this.flowLabels = ['Pseudo-Working', 'Task-Switching', 'Distracted', 'Soft Flow', 'Deep Flow'];
        this.speechClassNames = ['Erratic', 'Slow', 'Normal', 'Fast', 'Rapid'];
        this.taskClassNames = ['DEEP_WORK', 'SHALLOW_WORK', 'CREATIVE', 'ADMINISTRATIVE', 'COMMUNICATION'];
        this.screenClassNames = ['PRODUCTIVE_CODE', 'PRODUCTIVE_DOCS', 'COMMUNICATION', 'DISTRACTION', 'NEUTRAL'];
        this.demandMap = { 0: 0.9, 1: 0.2, 2: 0.7, 3: 0.3, 4: 0.5 };

        // Vision model type: 'finetuned' (4 classes) or 'pretrained_coco' (80 classes)
        this.visionModelType = 'finetuned';

        // Chrome Extension bridge — real tab data
        this._extensionTabData = null;
        this._extensionConnected = false;

        // Live audio level (0-1) updated every frame
        this.currentAudioLevel = 0;
        this._audioLevelAnimFrame = null;

        // Last screen classification result (for UI)
        this.lastScreenResult = null;
    }

    /** Initialize all components */
    async init(demoMode = true) {
        this.demoMode = demoMode;
        this.audioExtractor = new AudioExtractor();
        this.visionPreprocessor = new VisionPreprocessor();
        this.nlpTokenizer = new NLPTokenizer();
        this.screenClassifier = new ScreenClassifier();

        // Listen for real tab data from Chrome Extension
        this._setupExtensionBridge();

        // Always try to load models — loadModels() will set demoMode=true
        // if models aren't available, or demoMode=false if they load successfully
        await this.loadModels();

        // Load screen classifier (non-blocking — it's optional)
        this.screenClassifier.load().then(ok => {
            if (ok) console.log('✅ Screen classifier ready');
        });

        await this.nlpTokenizer.loadVocab();
        return true;
    }

    /** Set up listener for Chrome Extension tab data */
    _setupExtensionBridge() {
        window.addEventListener('message', (e) => {
            // Ensure message is coming from our window and has the right type
            if (e.source === window && e.data && e.data.type === 'ANI_TAB_DATA') {
                this._extensionTabData = e.data.data;
                this._extensionConnected = true;
                console.log('[ANI] Real tab data from extension:', e.data.data);
            }
        });

        // Request tab data immediately via postMessage
        window.postMessage({ type: 'ANI_REQUEST_TABS' }, '*');

        // Request periodically
        setInterval(() => {
            window.postMessage({ type: 'ANI_REQUEST_TABS' }, '*');
        }, 5000);
    }

    /** Get whether extension is providing real tab data */
    get extensionConnected() { return this._extensionConnected; }

    /** Get real tab count or null if extension not connected */
    getRealTabCount() { return this._extensionTabData?.tabCount ?? null; }

    /** Load all ONNX models */
    async loadModels() {
        try {
            if (typeof ort === 'undefined') {
                await this._loadScript('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');
            }

            // Detect vision model type from class mapping
            try {
                const mappingResp = await fetch('../models/vision_class_mapping.json');
                if (mappingResp.ok) {
                    const mapping = await mappingResp.json();
                    this.visionModelType = mapping.mode || 'finetuned';
                    console.log(`[ANI] Vision model type: ${this.visionModelType}`);
                }
            } catch (e) {
                console.warn('Could not load vision class mapping, defaulting to finetuned');
            }

            const modelPaths = {
                vision: '../models/desk_distraction_v1.onnx',
                audio: '../models/speech_classifier.onnx',
                // NLP: replaced DistilBERT (256MB) with lightweight keyword classifier (0 model files)
                meta: '../models/meta_flow_classifier.onnx',
            };

            for (const [name, path] of Object.entries(modelPaths)) {
                try {
                    this.models[name] = await ort.InferenceSession.create(path);
                    console.log(`✅ Model loaded: ${name}`);
                } catch (e) {
                    console.warn(`⚠️ Model not found: ${name} (${path})`);
                }
            }

            this.modelsLoaded = Object.keys(this.models).length >= 2; // At least audio + meta
            if (!this.modelsLoaded) {
                console.warn('Not enough models loaded, falling back to demo mode');
                this.demoMode = true;
            } else {
                this.demoMode = false;
                console.log(`✅ ${Object.keys(this.models).length} models loaded — REAL inference mode`);
            }
            return this.modelsLoaded;
        } catch (e) {
            console.warn('ONNX Runtime not available, using demo mode:', e);
            this.demoMode = true;
            return false;
        }
    }

    /** Dynamically load a script */
    _loadScript(src) {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = src;
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    /** Enable microphone */
    async enableMicrophone() {
        return await this.audioExtractor.init();
    }

    /** Enable webcam for desk/phone detection */
    async enableWebcam() {
        return await this.visionPreprocessor.initWebcam();
    }

    /** Enable screen capture */
    async enableScreenCapture() {
        return await this.visionPreprocessor.initScreenCapture();
    }

    /** Get vision preprocessor for webcam preview drawing */
    getVisionPreprocessor() {
        return this.visionPreprocessor;
    }

    /** Run full inference pipeline */
    async analyze(taskText = '') {
        if (this.demoMode) {
            return this._demoInference(taskText);
        }
        return this._fullInference(taskText);
    }

    /** Demo inference — uses real data sources when available, simulates the rest */
    _demoInference(taskText) {
        // ─── Vision: Webcam YOLO > Extension tabs > Screen > Simulated ──
        let visionFeatures;
        let visionSource = 'simulated';

        if (this.visionPreprocessor.isActive && this.models.vision) {
            // BEST: Real webcam + trained YOLO model
            try {
                const tensors = this.visionPreprocessor.captureAll();
                if (tensors.length > 0) {
                    // Run YOLO synchronously is not possible, we'll handle async later
                    // For now use webcam heuristic + last detections
                    visionFeatures = this.visionPreprocessor.extractDemoFeatures();
                    visionSource = tensors[0].source;
                } else {
                    visionFeatures = this.visionPreprocessor.extractDemoFeatures();
                    visionSource = 'heuristic';
                }
            } catch (e) {
                visionFeatures = this.visionPreprocessor.extractDemoFeatures();
                visionSource = 'heuristic';
            }
        } else if (this.visionPreprocessor.isActive) {
            // Webcam/Screen active but no YOLO model
            visionFeatures = this.visionPreprocessor.extractDemoFeatures();
            visionSource = (this.visionPreprocessor.webcamVideo) ? 'webcam' : 'screen';
        } else if (this._extensionConnected && this._extensionTabData) {
            // REAL tab data from Chrome Extension
            const ext = this._extensionTabData;
            visionFeatures = {
                tab_count_norm: ext.tabCountNorm || Math.min(ext.tabCount / 30, 1.0),
                phone_visible: 0,
                distraction_count_norm: Math.min((ext.distractionTabs || 0) / 5, 1.0),
                focus_ratio: ext.tabCount > 0 ? Math.max(0.1, 1 - (ext.distractionTabs / ext.tabCount)) : 0.5,
                detections: [],
            };
            visionSource = 'extension';
        } else {
            visionFeatures = this.visionPreprocessor.extractDemoFeatures();
            visionSource = 'simulated';
        }

        // ─── Screen Productivity: MobileNetV3 classification ─────
        let screenResult = null;
        if (this.visionPreprocessor.screenVideo && this.visionPreprocessor.screenVideo.videoWidth > 0) {
            // Use screen classifier (sync fallback in demo mode)
            screenResult = this.screenClassifier._fallbackClassify(this.visionPreprocessor.screenVideo);
            this.lastScreenResult = screenResult;

            // Override focus_ratio with screen productivity score (Option A)
            visionFeatures.focus_ratio = screenResult.productivityScore;
            visionSource = visionSource === 'simulated' ? 'screen-classifier' : visionSource + ' + screen';
        }

        // ─── Audio: Use real mic data if recording ───────────────
        let audioMeta = { speech_class: 2, speech_confidence: 0.8, wpm_norm: 0.55, fluency_score: 0.7 };
        let audioSource = 'simulated';

        if (this.audioExtractor.isRecording) {
            const rawFeatures = this.audioExtractor.extractFeatures();
            const voiceState = this.audioExtractor.getVoiceState();

            // Update live audio level for visualizer
            this.currentAudioLevel = voiceState.energyValue;

            // Map Voice State to speech class for the meta-classifier
            // 0=Erratic → Stressed tone + high energy
            // 1=Slow → Silent/Quiet energy
            // 2=Normal → Active energy + Neutral/Calm tone
            // 3=Fast → Active energy + Animated tone
            // 4=Rapid → Energized energy + Stressed tone
            let speechClass = 2; // Normal
            if (!voiceState.isActive) {
                speechClass = 1; // Silent
            } else if (voiceState.energyLevel === 'Energized' && voiceState.tone === 'Stressed') {
                speechClass = 4; // Rapid/Intense
            } else if (voiceState.energyLevel === 'Active' && (voiceState.tone === 'Animated' || voiceState.tone === 'Stressed')) {
                speechClass = 3; // Fast/Engaged
            } else if (voiceState.tone === 'Stressed' && voiceState.energyLevel !== 'Quiet') {
                speechClass = 0; // Erratic
            }

            audioMeta = {
                speech_class: speechClass,
                speech_confidence: Math.min(0.95, 0.5 + voiceState.rms * 5),
                wpm_norm: voiceState.energyValue,  // Energy as proxy for speech intensity
                fluency_score: voiceState.activityPercent / 100,
                // Voice State metadata for UI
                energyLevel: voiceState.energyLevel,
                tone: voiceState.tone,
                activityPercent: voiceState.activityPercent,
            };
            audioSource = 'microphone';
        } else {
            this.currentAudioLevel = 0;
        }

        // ─── NLP: Always real (keyword classifier runs locally) ──
        let nlpResult = { taskClass: 0, cognitiveDemand: 0.5, confidence: 0.5, className: 'UNKNOWN' };
        if (taskText.trim().length > 0) {
            nlpResult = this.nlpTokenizer.classifyDemo(taskText);
        }

        // Fuse into 11-feature vector
        const fusedVector = [
            visionFeatures.tab_count_norm,
            visionFeatures.phone_visible,
            visionFeatures.distraction_count_norm,
            visionFeatures.focus_ratio,
            audioMeta.speech_class,
            audioMeta.speech_confidence,
            audioMeta.wpm_norm,
            audioMeta.fluency_score,
            nlpResult.taskClass,
            nlpResult.cognitiveDemand,
            nlpResult.confidence,
        ];

        // Demo meta-classifier — weighted scoring
        const flowScores = this._demoMetaClassifier(fusedVector);
        const predictedClass = flowScores.indexOf(Math.max(...flowScores));

        // Build result with data source tags
        const extData = this._extensionTabData;
        return {
            flowState: predictedClass,
            flowStateName: this.flowStateNames[predictedClass],
            flowLabel: this.flowLabels[predictedClass],
            flowEmoji: this.flowEmojis[predictedClass],
            workQuality: flowScores[3] + flowScores[4],
            probabilities: flowScores,
            confidence: Math.max(...flowScores),
            vision: {
                tabCount: extData ? extData.tabCount : Math.round(visionFeatures.tab_count_norm * 30),
                phoneVisible: visionFeatures.phone_visible ? 'Yes' : 'No',
                distractions: extData ? (extData.distractionTabs || 0) : Math.round(visionFeatures.distraction_count_norm * 5),
                focusRatio: (visionFeatures.focus_ratio * 100).toFixed(0) + '%',
                features: visionFeatures,
                source: visionSource,
                extensionConnected: !!this._extensionConnected,
                tabCategories: extData ? extData.categories : null,
                activeTabTitle: extData && extData.activeTab ? extData.activeTab.title : null,
                activeTabUrl: extData && extData.activeTab ? extData.activeTab.url : null,
                productivityScore: extData ? extData.productivityScore : null,
                switchRate: extData ? extData.switchRate : null,
            },
            screen: screenResult ? {
                className: screenResult.className,
                confidence: (screenResult.confidence * 100).toFixed(0) + '%',
                productivityScore: (screenResult.productivityScore * 100).toFixed(0) + '%',
                source: screenResult.source,
            } : null,
            audio: {
                speechClass: this.speechClassNames[audioMeta.speech_class],
                energyLevel: audioMeta.energyLevel || this.speechClassNames[audioMeta.speech_class],
                tone: audioMeta.tone || 'Neutral',
                activity: audioMeta.activityPercent !== undefined ? audioMeta.activityPercent + '%' : (audioMeta.fluency_score * 100).toFixed(0) + '%',
                confidence: (audioMeta.speech_confidence * 100).toFixed(0) + '%',
                features: audioMeta,
                source: audioSource,
            },
            nlp: {
                taskType: nlpResult.className,
                demand: (nlpResult.cognitiveDemand * 100).toFixed(0) + '%',
                confidence: (nlpResult.confidence * 100).toFixed(0) + '%',
                features: nlpResult,
                source: 'local-nlp',
            },
            featureImportances: this._computeFeatureImportance(fusedVector, predictedClass),
            dataSources: { vision: visionSource, audio: audioSource, nlp: 'local-nlp' },
            timestamp: Date.now(),
        };
    }

    /** Demo meta-classifier — uses interpretable rules derived from training data patterns */
    _demoMetaClassifier(features) {
        const [tabNorm, phoneVis, distNorm, focusR, spClass, spConf, wpmN, fluency, taskCls, cogDemand, taskConf] = features;
        const scores = new Float32Array(5);

        // PSEUDO_WORKING: many tabs, low demand, low fluency
        scores[0] = (tabNorm * 0.3 + (1 - cogDemand) * 0.3 + (1 - fluency) * 0.2 + (1 - focusR) * 0.2) * 0.8;

        // TASK_SWITCHING: many tabs, fast speech, admin tasks
        scores[1] = (tabNorm * 0.35 + wpmN * 0.25 + (taskCls === 3 ? 0.3 : 0.05) + distNorm * 0.1) * 0.8;

        // DISTRACTED: phone visible, many distractions, low focus
        scores[2] = (phoneVis * 0.35 + distNorm * 0.3 + (1 - focusR) * 0.25 + (1 - taskConf) * 0.1) * 0.9;

        // SOFT_FLOW: moderate focus, normal speech, decent demand
        scores[3] = (focusR * 0.3 + fluency * 0.2 + cogDemand * 0.2 + (1 - distNorm) * 0.15 + (1 - phoneVis) * 0.15) * 0.85;

        // DEEP_FLOW: high focus, low distractions, high demand, steady speech
        scores[4] = (focusR * 0.25 + (1 - tabNorm) * 0.15 + (1 - phoneVis) * 0.15 + (1 - distNorm) * 0.15 + cogDemand * 0.15 + fluency * 0.15) * 0.9;

        // Add small noise
        for (let i = 0; i < 5; i++) scores[i] += Math.random() * 0.05;

        // Normalize to probabilities
        const total = scores.reduce((a, b) => a + b, 0);
        for (let i = 0; i < 5; i++) scores[i] = total > 0 ? scores[i] / total : 0.2;

        return Array.from(scores);
    }

    /** Compute which features drove the prediction */
    _computeFeatureImportance(features, predictedClass) {
        const names = ['Tabs', 'Phone', 'Distractions', 'Focus', 'Speech',
                       'Speech Conf', 'Speed', 'Fluency', 'Task Type', 'Demand', 'Task Conf'];

        // Simple importance based on deviation from "neutral" (0.5)
        const importances = features.map((f, i) => ({
            name: names[i],
            value: features[i],
            importance: Math.abs(f - 0.5) * (i < 4 ? 1.2 : i < 8 ? 1.0 : 0.8),
        }));

        return importances.sort((a, b) => b.importance - a.importance).slice(0, 5);
    }

    /** Full ONNX-based inference */
    async _fullInference(taskText) {
        // If extension connected, use active tab title as NLP context
        const extData = this._extensionTabData;
        const nlpText = taskText || (extData?.activeTab?.title) || '';

        // Run all modality models
        const visionResult = await this._runVisionModel();
        const audioResult = await this._runAudioModel();
        const nlpResult = await this._runNLPModel(nlpText);

        // Run screen classifier if screen capture is active
        let screenResult = null;
        if (this.visionPreprocessor.screenVideo && this.visionPreprocessor.screenVideo.videoWidth > 0) {
            screenResult = await this.screenClassifier.classify(this.visionPreprocessor.screenVideo);
            this.lastScreenResult = screenResult;

            // Option A: Override focus_ratio with screen productivity score
            visionResult.focus_ratio = screenResult.productivityScore;
            console.log(`[Screen] ${screenResult.className} (${(screenResult.confidence * 100).toFixed(0)}%) → productivity=${(screenResult.productivityScore * 100).toFixed(0)}%`);
        }

        // Enrich vision with extension tab data when available
        if (extData && extData.extensionConnected) {
            visionResult.tab_count_norm = extData.tabCountNorm || visionResult.tab_count_norm;
            visionResult.distraction_count_norm = extData.distractionScore || visionResult.distraction_count_norm;
            // Only use extension productivity if screen classifier didn't override
            if (!screenResult) {
                visionResult.focus_ratio = Math.max(visionResult.focus_ratio, extData.productivityScore || 0);
            }
            // Extension provides real source
            if (visionResult.source === 'no-webcam' || visionResult.source === 'no-model') {
                visionResult.source = 'Chrome Extension';
            } else {
                visionResult.source = visionResult.source + ' + Extension';
            }
        }

        // Fuse and run meta-classifier
        const fusedVector = new Float32Array([
            visionResult.tab_count_norm, visionResult.phone_visible,
            visionResult.distraction_count_norm, visionResult.focus_ratio,
            audioResult.speech_class, audioResult.speech_confidence,
            audioResult.wpm_norm, audioResult.fluency_score,
            nlpResult.task_class, nlpResult.cognitive_demand, nlpResult.confidence,
        ]);

        let flowState = 2, probs = [0.2, 0.2, 0.2, 0.2, 0.2];

        if (this.models.meta) {
            try {
                const metaTensor = new ort.Tensor('float32', fusedVector, [1, 11]);
                const metaResult = await this.models.meta.run({ input: metaTensor });
                
                // Parse label — sklearn RF exports as int64 tensor named 'label' or 'output_label'
                const labelKey = Object.keys(metaResult).find(k => k.includes('label') || k === 'output_label');
                const label = labelKey ? Number(metaResult[labelKey].data[0]) : 0;
                flowState = label;
                
                // Parse probabilities — sklearn RF exports as sequence of maps
                // which onnxruntime-web may expose differently. Try multiple approaches.
                const probKey = Object.keys(metaResult).find(k => k.includes('probabilities') || k.includes('probability'));
                if (probKey) {
                    try {
                        // Approach 1: If it's a standard tensor
                        const probData = metaResult[probKey].data;
                        if (probData && probData.length >= 5) {
                            probs = Array.from(probData).slice(0, 5);
                        }
                    } catch (probErr) {
                        // Approach 2: Sequence of maps — not directly readable
                        // Fall back to setting high confidence on the predicted class
                        console.warn('[Meta] Probability format not directly readable, using label-based estimation');
                        probs = [0.05, 0.05, 0.05, 0.05, 0.05];
                        probs[flowState] = 0.8;
                    }
                } else {
                    // No probability output — estimate from label
                    probs = [0.05, 0.05, 0.05, 0.05, 0.05];
                    probs[flowState] = 0.8;
                }
                
                console.log(`[Meta] Flow state=${flowState} (${this.flowStateNames[flowState]}), probs=[${probs.map(p => (p*100).toFixed(0)+'%').join(',')}]`);
            } catch (e) {
                console.warn('Meta model inference failed:', e);
                const demoScores = this._demoMetaClassifier(Array.from(fusedVector));
                flowState = demoScores.indexOf(Math.max(...demoScores));
                probs = demoScores;
            }
        } else {
            const demoScores = this._demoMetaClassifier(Array.from(fusedVector));
            flowState = demoScores.indexOf(Math.max(...demoScores));
            probs = demoScores;
        }

        // Determine data sources
        const visionSrc = visionResult.source || (this.models.vision ? 'ONNX Model' : 'demo');
        const audioSrc = audioResult.source || (this.models.audio ? 'ONNX Model' : 'demo');
        const nlpSrc = nlpResult.source || (this.models.nlp ? 'ONNX Model' : 'demo');
        const metaSrc = this.models.meta ? 'ONNX Model' : 'demo';

        return {
            flowState,
            flowStateName: this.flowStateNames[flowState],
            flowLabel: this.flowLabels[flowState],
            flowEmoji: this.flowEmojis[flowState],
            workQuality: probs[3] + probs[4],
            probabilities: probs,
            confidence: Math.max(...probs),
            vision: {
                tabCount: extData?.tabCount || Math.round(visionResult.tab_count_norm * 30),
                phoneVisible: visionResult.phone_visible ? 'Yes' : 'No',
                distractions: extData ? (extData.categories?.distraction || 0) + (extData.categories?.news || 0) : Math.round(visionResult.distraction_count_norm * 5),
                focusRatio: (visionResult.focus_ratio * 100).toFixed(0) + '%',
                features: visionResult,
                detections: visionResult.detections || [],
                source: visionSrc,
                // Extension tab analysis
                extensionConnected: !!extData?.extensionConnected,
                tabCategories: extData?.categories || null,
                productivityScore: extData?.productivityScore || null,
                distractionScore: extData?.distractionScore || null,
                switchRate: extData?.switchRate || 0,
                activeTabTitle: extData?.activeTab?.title || null,
                activeTabCategory: extData?.activeTab?.category || null,
                allTabs: extData?.tabs || [],
            },
            screen: screenResult ? {
                className: screenResult.className,
                confidence: (screenResult.confidence * 100).toFixed(0) + '%',
                productivityScore: (screenResult.productivityScore * 100).toFixed(0) + '%',
                rawProbs: screenResult.rawProbs,
                source: screenResult.source,
            } : null,
            audio: {
                speechClass: this.speechClassNames[audioResult.speech_class] || 'Normal',
                energyLevel: audioResult.energyLevel || this.speechClassNames[audioResult.speech_class] || 'Normal',
                tone: audioResult.tone || 'Neutral',
                activity: audioResult.activityPercent !== undefined ? audioResult.activityPercent + '%' : (audioResult.fluency_score * 100).toFixed(0) + '%',
                confidence: (audioResult.speech_confidence * 100).toFixed(0) + '%',
                features: audioResult,
                classProbs: audioResult.class_probs || null,
                source: audioSrc,
            },
            nlp: {
                taskType: this.taskClassNames[nlpResult.task_class] || 'UNKNOWN',
                demand: (nlpResult.cognitive_demand * 100).toFixed(0) + '%',
                confidence: (nlpResult.confidence * 100).toFixed(0) + '%',
                features: nlpResult,
                classProbs: nlpResult.class_probs || null,
                source: nlpSrc,
                analyzedText: nlpText || null,
            },
            meta: {
                flowState: this.flowStateNames[flowState],
                classProbs: probs,
                source: metaSrc,
                fusedVector: Array.from(fusedVector),
            },
            featureImportances: this._computeFeatureImportance(Array.from(fusedVector), flowState),
            dataSources: { vision: visionSrc, audio: audioSrc, nlp: nlpSrc, meta: metaSrc, screen: screenResult?.source || 'none' },
            extensionConnected: !!extData?.extensionConnected,
            timestamp: Date.now(),
        };
    }

    /** Run vision model and extract 4 features */
    async _runVisionModel() {
        const defaults = { tab_count_norm: 0.3, phone_visible: 0, distraction_count_norm: 0.2, focus_ratio: 0.6, detections: [], source: 'no-webcam' };
        
        if (!this.models.vision) {
            const demo = this.visionPreprocessor.extractDemoFeatures();
            return { ...demo, source: 'no-model', detections: demo.detections || [] };
        }

        try {
            const tensors = this.visionPreprocessor.captureAll();
            if (!tensors || tensors.length === 0) {
                return { ...defaults, source: this.visionPreprocessor.isActive ? 'capture-failed' : 'no-webcam' };
            }

            const inputNames = this.models.vision.inputNames || ['images'];
            const inputName = inputNames[0];
            const isCoco = this.visionModelType === 'pretrained_coco';

            let combinedDetections = [];
            let phoneVisible = 0, monitorCount = 0, workToolCount = 0, distractionCount = 0;
            
            for (const { source, tensor: imageData } of tensors) {
                const tensor = new ort.Tensor('float32', imageData, [1, 3, 640, 640]);
                const feeds = {};
                feeds[inputName] = tensor;
                const result = await this.models.vision.run(feeds);
                
                const output = result[Object.keys(result)[0]];
                const parsed = this.visionPreprocessor.parseDetections(output, isCoco, source);
                
                combinedDetections = combinedDetections.concat(parsed.detections);
                
                phoneVisible = Math.max(phoneVisible, parsed.phone_visible);
                for (const det of parsed.detections) {
                    switch(det.classId) {
                        case 1: monitorCount++; break;
                        case 2: workToolCount++; break;
                        case 3: distractionCount++; break;
                    }
                }
            }

            const tab_count_norm = Math.min(monitorCount / 3, 1.0);
            const distraction_count_norm = Math.min(distractionCount / 5, 1.0);
            const focus_ratio = monitorCount > 0 || workToolCount > 0
                ? Math.max(0.3, 1 - (distractionCount / Math.max(1, monitorCount + workToolCount + distractionCount)))
                : 0.5;

            console.log(`[YOLO] ${combinedDetections.length} detections: ${combinedDetections.map(d => d.className + ':' + (d.confidence*100).toFixed(0) + '%').join(', ') || 'none'}`);
            
            return {
                tab_count_norm,
                phone_visible: phoneVisible,
                distraction_count_norm,
                focus_ratio,
                detections: combinedDetections,
                source: 'ONNX Model',
            };
        } catch (e) {
            console.warn('Vision model inference failed:', e);
            return { ...defaults, source: 'error' };
        }
    }

    /** Run audio model and extract 4 meta features */
    async _runAudioModel() {
        const defaults = { speech_class: 2, speech_confidence: 0.7, wpm_norm: 0.55, fluency_score: 0.7, class_probs: null, source: 'no-mic' };
        
        if (!this.audioExtractor.isRecording) return defaults;
        
        const rawFeatures = this.audioExtractor.extractFeatures();
        
        // Update live audio level for visualizer
        this.currentAudioLevel = Math.min(1, (rawFeatures[45] || 0) * 10);
        
        // Always get voice state for UI display
        const voiceState = this.audioExtractor.getVoiceState();
        
        if (!this.models.audio) {
            // Use voice state for classification instead of WPM
            const rms = rawFeatures[45];
            const silenceRatio = rawFeatures[51];
            const energyNorm = rawFeatures[49]; // Voice energy normalized
            
            // Speech gate: if RMS is very low, no one is really speaking
            let speechClass = 2;
            if (rms < 0.008) speechClass = 1; // Silent
            else if (rms > 0.08) speechClass = 3; // Energized/Active
            else if (silenceRatio > 0.5 && rms < 0.02) speechClass = 0; // Erratic/Low
            
            return {
                speech_class: speechClass,
                speech_confidence: 0.6 + Math.random() * 0.3,
                wpm_norm: energyNorm,
                fluency_score: 1 - silenceRatio,
                class_probs: null,
                source: 'no-model',
                energyLevel: voiceState.energyLevel,
                tone: voiceState.tone,
                activityPercent: voiceState.activityPercent,
            };
        }

        try {
            const tensor = new ort.Tensor('float32', rawFeatures, [1, 52]);
            const result = await this.models.audio.run({ input: tensor });
            
            let label = result.label ? Number(result.label.data[0]) : 2;
            const probsData = result.probabilities ? Array.from(result.probabilities.data) : null;
            const confidence = probsData ? Math.max(...probsData) : 0.7;
            
            const rms = rawFeatures[45];
            const energyNorm = rawFeatures[49];
            
            // 🔥 HARD GATE: Override AI classification using reliable energy level.
            // If RMS is too low for real speech, force Silent regardless of model output.
            if (rms < 0.008) {
                label = 1; // Silent
            } else if (rms > 0.08 && label < 3) {
                label = 3; // At least "Fast/Energized" if loud
            }

            console.log(`[XGBoost] Speech class=${label} (${this.speechClassNames[label]}), energy=${energyNorm.toFixed(2)}, confidence=${(confidence*100).toFixed(0)}%`);
            
            return {
                speech_class: label,
                speech_confidence: confidence,
                wpm_norm: energyNorm,
                fluency_score: 1 - rawFeatures[51],
                class_probs: probsData,
                source: 'ONNX Model',
                energyLevel: voiceState.energyLevel,
                tone: voiceState.tone,
                activityPercent: voiceState.activityPercent,
            };
        } catch (e) {
            console.warn('Audio model inference failed:', e);
            return { ...defaults, source: 'error', energyLevel: voiceState.energyLevel, tone: voiceState.tone, activityPercent: voiceState.activityPercent };
        }
    }

    /** Run NLP classifier and extract 3 features */
    async _runNLPModel(taskText) {
        const defaults = { task_class: 0, cognitive_demand: 0.5, confidence: 0.5, class_probs: null, source: 'no-text' };
        if (!taskText || taskText.trim().length === 0) return defaults;

        // Use the lightweight keyword + URL + regex classifier (replaced DistilBERT 256MB)
        const extData = this._extensionTabData;
        const url = extData?.activeTab?.url || '';
        const result = this.nlpTokenizer.classify(taskText, url);

        console.log(`[TaskClassifier] "${taskText.substring(0,40)}" → ${this.taskClassNames[result.taskClass]} (${(result.confidence*100).toFixed(0)}%)`);

        return {
            task_class: result.taskClass,
            cognitive_demand: result.cognitiveDemand,
            confidence: result.confidence,
            class_probs: result.probabilities,
            source: 'Keyword Engine',
        };
    }

    /** Stop all inputs */
    stop() {
        if (this.audioExtractor) this.audioExtractor.stop();
        if (this.visionPreprocessor) this.visionPreprocessor.stop();
    }

    /** Get audio frequency data for visualization */
    getAudioFrequencyData() {
        return this.audioExtractor ? this.audioExtractor.getFrequencyData() : new Uint8Array(0);
    }
}

window.InferencePipeline = InferencePipeline;
