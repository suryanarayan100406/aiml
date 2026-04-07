/**
 * InferencePipeline — ONNX Runtime Web orchestrator for all 4 models.
 * Runs: Vision → Audio → NLP → Meta-classifier fusion.
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
        this.flowStateNames = ['PSEUDO_WORKING', 'TASK_SWITCHING', 'DISTRACTED', 'SOFT_FLOW', 'DEEP_FLOW'];
        this.flowEmojis = ['🔴', '🟠', '🟡', '🟢', '🟣'];
        this.flowLabels = ['Pseudo-Working', 'Task-Switching', 'Distracted', 'Soft Flow', 'Deep Flow'];
        this.speechClassNames = ['Erratic', 'Slow', 'Normal', 'Fast', 'Rapid'];
        this.taskClassNames = ['DEEP_WORK', 'SHALLOW_WORK', 'CREATIVE', 'ADMINISTRATIVE', 'COMMUNICATION'];
        this.demandMap = { 0: 0.9, 1: 0.2, 2: 0.7, 3: 0.3, 4: 0.5 };

        // Vision model type: 'finetuned' (4 classes) or 'pretrained_coco' (80 classes)
        this.visionModelType = 'finetuned';

        // Chrome Extension bridge — real tab data
        this._extensionTabData = null;
        this._extensionConnected = false;

        // Live audio level (0-1) updated every frame
        this.currentAudioLevel = 0;
        this._audioLevelAnimFrame = null;
    }

    /** Initialize all components */
    async init(demoMode = true) {
        this.demoMode = demoMode;
        this.audioExtractor = new AudioExtractor();
        this.visionPreprocessor = new VisionPreprocessor();
        this.nlpTokenizer = new NLPTokenizer();

        // Listen for real tab data from Chrome Extension
        this._setupExtensionBridge();

        // Always try to load models — loadModels() will set demoMode=true
        // if models aren't available, or demoMode=false if they load successfully
        await this.loadModels();

        await this.nlpTokenizer.loadVocab();
        return true;
    }

    /** Set up listener for Chrome Extension tab data */
    _setupExtensionBridge() {
        window.addEventListener('ani-tab-data', (e) => {
            this._extensionTabData = e.detail;
            this._extensionConnected = true;
            console.log('[ANI] Real tab data from extension:', e.detail);
        });

        // Request tab data immediately
        window.dispatchEvent(new CustomEvent('ani-request-tabs'));

        // Request periodically
        setInterval(() => {
            window.dispatchEvent(new CustomEvent('ani-request-tabs'));
        }, 10000);
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
                nlp: '../models/task_nlp_classifier.onnx',
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

        if (this.visionPreprocessor.mode === 'webcam' && this.models.vision) {
            // BEST: Real webcam + trained YOLO model
            try {
                const imageData = this.visionPreprocessor.captureAndPreprocess();
                if (imageData) {
                    // Run YOLO synchronously is not possible, we'll handle async later
                    // For now use webcam heuristic + last detections
                    visionFeatures = this.visionPreprocessor.extractDemoFeatures();
                    visionSource = 'webcam';
                } else {
                    visionFeatures = this.visionPreprocessor.extractDemoFeatures();
                    visionSource = 'webcam-heuristic';
                }
            } catch (e) {
                visionFeatures = this.visionPreprocessor.extractDemoFeatures();
                visionSource = 'webcam-heuristic';
            }
        } else if (this.visionPreprocessor.mode === 'webcam') {
            // Webcam active but no YOLO model — still show webcam feed
            visionFeatures = this.visionPreprocessor.extractDemoFeatures();
            visionSource = 'webcam';
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
            visionSource = this.visionPreprocessor.mode === 'screen' ? 'screen-capture' : 'simulated';
        }

        // ─── Audio: Use real mic data if recording ───────────────
        let audioMeta = { speech_class: 2, speech_confidence: 0.8, wpm_norm: 0.55, fluency_score: 0.7 };
        let audioSource = 'simulated';

        if (this.audioExtractor.isRecording) {
            const rawFeatures = this.audioExtractor.extractFeatures();
            const rms = rawFeatures[45];
            const silenceRatio = rawFeatures[51];
            const wpm = rawFeatures[49];
            const zcr = rawFeatures[44];

            // Update live audio level for visualizer
            this.currentAudioLevel = Math.min(1, rms * 10);

            // Map raw features to meta features
            let speechClass = 2; // Normal
            if (silenceRatio > 0.8) speechClass = 1; // Mostly silent / slow
            else if (wpm > 200) speechClass = 4; // Rapid
            else if (rms > 0.08 && wpm > 160) speechClass = 3; // Fast energized
            else if (rms > 0.04 && zcr > 0.15) speechClass = 0; // Erratic

            audioMeta = {
                speech_class: speechClass,
                speech_confidence: Math.min(0.95, 0.5 + rms * 5),
                wpm_norm: Math.min(1, wpm / 220),
                fluency_score: Math.max(0, 1 - silenceRatio),
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
            },
            audio: {
                speechClass: this.speechClassNames[audioMeta.speech_class],
                wpm: Math.round(audioMeta.wpm_norm * 220),
                fluency: (audioMeta.fluency_score * 100).toFixed(0) + '%',
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
        // Run all 3 modality models
        const visionResult = await this._runVisionModel();
        const audioResult = await this._runAudioModel();
        const nlpResult = await this._runNLPModel(taskText);

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
                const label = metaResult.label ? metaResult.label.data[0] : 0;
                probs = metaResult.probabilities ? Array.from(metaResult.probabilities.data) : probs;
                flowState = Number(label);
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

        return {
            flowState,
            flowStateName: this.flowStateNames[flowState],
            flowLabel: this.flowLabels[flowState],
            flowEmoji: this.flowEmojis[flowState],
            workQuality: probs[3] + probs[4],
            probabilities: probs,
            confidence: Math.max(...probs),
            vision: {
                tabCount: Math.round(visionResult.tab_count_norm * 30),
                phoneVisible: visionResult.phone_visible ? 'Yes' : 'No',
                distractions: Math.round(visionResult.distraction_count_norm * 5),
                focusRatio: (visionResult.focus_ratio * 100).toFixed(0) + '%',
                features: visionResult,
            },
            audio: {
                speechClass: this.speechClassNames[audioResult.speech_class] || 'Normal',
                wpm: Math.round(audioResult.wpm_norm * 220),
                fluency: (audioResult.fluency_score * 100).toFixed(0) + '%',
                confidence: (audioResult.speech_confidence * 100).toFixed(0) + '%',
                features: audioResult,
            },
            nlp: {
                taskType: this.taskClassNames[nlpResult.task_class] || 'UNKNOWN',
                demand: (nlpResult.cognitive_demand * 100).toFixed(0) + '%',
                confidence: (nlpResult.confidence * 100).toFixed(0) + '%',
                features: nlpResult,
            },
            featureImportances: this._computeFeatureImportance(Array.from(fusedVector), flowState),
            timestamp: Date.now(),
        };
    }

    /** Run vision model and extract 4 features */
    async _runVisionModel() {
        const defaults = { tab_count_norm: 0.3, phone_visible: 0, distraction_count_norm: 0.2, focus_ratio: 0.6 };
        
        if (!this.models.vision) {
            // Use demo features from screen capture if available
            const demo = this.visionPreprocessor.extractDemoFeatures();
            return {
                tab_count_norm: demo.tab_count_norm,
                phone_visible: demo.phone_visible,
                distraction_count_norm: demo.distraction_count_norm,
                focus_ratio: demo.focus_ratio,
            };
        }

        try {
            const imageData = this.visionPreprocessor.captureAndPreprocess();
            if (!imageData) return defaults;

            // Get the correct input name from the model
            const inputNames = this.models.vision.inputNames || ['images'];
            const inputName = inputNames[0];
            const tensor = new ort.Tensor('float32', imageData, [1, 3, 640, 640]);
            const feeds = {};
            feeds[inputName] = tensor;
            const result = await this.models.vision.run(feeds);
            
            // Delegate detection parsing to VisionPreprocessor
            // which handles both YOLOv8 raw format [1, num_features, num_boxes]
            // and post-processed format [num_detections, 6]
            const output = result[Object.keys(result)[0]];
            const isCoco = this.visionModelType === 'pretrained_coco';
            const parsed = this.visionPreprocessor.parseDetections(output, isCoco);
            
            return {
                tab_count_norm: parsed.tab_count_norm,
                phone_visible: parsed.phone_visible,
                distraction_count_norm: parsed.distraction_count_norm,
                focus_ratio: parsed.focus_ratio,
            };
        } catch (e) {
            console.warn('Vision model inference failed:', e);
            return defaults;
        }
    }

    /** Run audio model and extract 4 meta features */
    async _runAudioModel() {
        const defaults = { speech_class: 2, speech_confidence: 0.7, wpm_norm: 0.55, fluency_score: 0.7 };
        
        if (!this.audioExtractor.isRecording) return defaults;
        
        const rawFeatures = this.audioExtractor.extractFeatures();
        
        if (!this.models.audio) {
            // Demo mode — derive meta features from raw audio
            const rms = rawFeatures[45];
            const silenceRatio = rawFeatures[51];
            const wpm = rawFeatures[49];
            
            let speechClass = 2;
            if (silenceRatio > 0.4) speechClass = 1;
            else if (wpm > 200) speechClass = 4;
            else if (rms > 0.08) speechClass = 3;
            else if (silenceRatio > 0.3 && rms < 0.03) speechClass = 0;
            
            return {
                speech_class: speechClass,
                speech_confidence: 0.6 + Math.random() * 0.3,
                wpm_norm: Math.min(1, wpm / 220),
                fluency_score: 1 - silenceRatio,
            };
        }

        try {
            const tensor = new ort.Tensor('float32', rawFeatures, [1, 52]);
            const result = await this.models.audio.run({ input: tensor });
            
            // Parse XGBoost output — label + probabilities
            const label = result.label ? Number(result.label.data[0]) : 2;
            const probsData = result.probabilities ? result.probabilities.data : null;
            const confidence = probsData ? Math.max(...Array.from(probsData)) : 0.7;
            
            return {
                speech_class: label,
                speech_confidence: confidence,
                wpm_norm: Math.min(1, rawFeatures[49] / 220),
                fluency_score: 1 - rawFeatures[51],
            };
        } catch (e) {
            console.warn('Audio model inference failed:', e);
            return defaults;
        }
    }

    /** Run NLP model and extract 3 features */
    async _runNLPModel(taskText) {
        const defaults = { task_class: 0, cognitive_demand: 0.5, confidence: 0.5 };
        if (!taskText || taskText.trim().length === 0) return defaults;

        if (!this.models.nlp) {
            // Use keyword-based demo classifier
            const demo = this.nlpTokenizer.classifyDemo(taskText);
            return {
                task_class: demo.taskClass,
                cognitive_demand: demo.cognitiveDemand,
                confidence: demo.confidence,
            };
        }

        try {
            const { inputIds, attentionMask } = this.nlpTokenizer.tokenize(taskText);
            const idsTensor = new ort.Tensor('int64', inputIds, [1, 128]);
            const maskTensor = new ort.Tensor('int64', attentionMask, [1, 128]);
            const result = await this.models.nlp.run({ input_ids: idsTensor, attention_mask: maskTensor });
            
            // Parse DistilBERT logits
            const logits = Array.from(result.logits.data);
            const maxLogit = Math.max(...logits);
            const expLogits = logits.map(l => Math.exp(l - maxLogit));
            const sumExp = expLogits.reduce((a, b) => a + b, 0);
            const probs = expLogits.map(e => e / sumExp);
            
            const predictedClass = probs.indexOf(Math.max(...probs));
            
            return {
                task_class: predictedClass,
                cognitive_demand: this.demandMap[predictedClass] || 0.5,
                confidence: Math.max(...probs),
            };
        } catch (e) {
            console.warn('NLP model inference failed:', e);
            const demo = this.nlpTokenizer.classifyDemo(taskText);
            return { task_class: demo.taskClass, cognitive_demand: demo.cognitiveDemand, confidence: demo.confidence };
        }
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
