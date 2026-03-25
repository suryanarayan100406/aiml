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
    }

    /** Initialize all components */
    async init(demoMode = true) {
        this.demoMode = demoMode;
        this.audioExtractor = new AudioExtractor();
        this.visionPreprocessor = new VisionPreprocessor();
        this.nlpTokenizer = new NLPTokenizer();

        if (!demoMode) {
            await this.loadModels();
        }

        await this.nlpTokenizer.loadVocab();
        return true;
    }

    /** Load all ONNX models */
    async loadModels() {
        try {
            if (typeof ort === 'undefined') {
                // Try loading ONNX Runtime Web
                await this._loadScript('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');
            }

            const modelPaths = {
                vision: 'models/desk_distraction_v1.onnx',
                audio: 'models/speech_classifier.onnx',
                nlp: 'models/task_nlp_classifier.onnx',
                meta: 'models/meta_flow_classifier.onnx',
            };

            for (const [name, path] of Object.entries(modelPaths)) {
                try {
                    this.models[name] = await ort.InferenceSession.create(path);
                    console.log(`✅ Model loaded: ${name}`);
                } catch (e) {
                    console.warn(`⚠️ Model not found: ${name} (${path})`);
                }
            }

            this.modelsLoaded = Object.keys(this.models).length === 4;
            if (!this.modelsLoaded) {
                console.warn('Not all models loaded, falling back to demo mode');
                this.demoMode = true;
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

    /** Enable screen capture */
    async enableScreenCapture() {
        return await this.visionPreprocessor.initScreenCapture();
    }

    /** Run full inference pipeline */
    async analyze(taskText = '') {
        if (this.demoMode) {
            return this._demoInference(taskText);
        }
        return this._fullInference(taskText);
    }

    /** Demo inference — simulated but realistic */
    _demoInference(taskText) {
        // Vision features
        const visionFeatures = this.visionPreprocessor.extractDemoFeatures();

        // Audio features
        let audioMeta = { speech_class: 2, speech_confidence: 0.8, wpm_norm: 0.55, fluency_score: 0.7 };
        if (this.audioExtractor.isRecording) {
            const rawFeatures = this.audioExtractor.extractFeatures();
            const rms = rawFeatures[45];
            const silenceRatio = rawFeatures[51];
            const wpm = rawFeatures[49];

            // Map raw features to meta features
            let speechClass = 2; // Normal
            if (silenceRatio > 0.4) speechClass = 1; // Slow
            else if (wpm > 200) speechClass = 4; // Rapid
            else if (rms > 0.08) speechClass = 3; // Fast energized

            audioMeta = {
                speech_class: speechClass,
                speech_confidence: 0.6 + Math.random() * 0.3,
                wpm_norm: Math.min(1, wpm / 220),
                fluency_score: 1 - silenceRatio,
            };
        }

        // NLP features
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

        return {
            flowState: predictedClass,
            flowStateName: this.flowStateNames[predictedClass],
            flowLabel: this.flowLabels[predictedClass],
            flowEmoji: this.flowEmojis[predictedClass],
            workQuality: flowScores[3] + flowScores[4], // soft + deep flow probability
            probabilities: flowScores,
            confidence: Math.max(...flowScores),
            vision: {
                tabCount: Math.round(visionFeatures.tab_count_norm * 30),
                phoneVisible: visionFeatures.phone_visible ? 'Yes' : 'No',
                distractions: Math.round(visionFeatures.distraction_count_norm * 5),
                focusRatio: (visionFeatures.focus_ratio * 100).toFixed(0) + '%',
                features: visionFeatures,
            },
            audio: {
                speechClass: ['Erratic', 'Slow', 'Normal', 'Fast', 'Rapid'][audioMeta.speech_class],
                wpm: Math.round(audioMeta.wpm_norm * 220),
                fluency: (audioMeta.fluency_score * 100).toFixed(0) + '%',
                confidence: (audioMeta.speech_confidence * 100).toFixed(0) + '%',
                features: audioMeta,
            },
            nlp: {
                taskType: nlpResult.className,
                demand: (nlpResult.cognitiveDemand * 100).toFixed(0) + '%',
                confidence: (nlpResult.confidence * 100).toFixed(0) + '%',
                features: nlpResult,
            },
            featureImportances: this._computeFeatureImportance(fusedVector, predictedClass),
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
        const visionFeatures = await this._runVisionModel();
        const audioFeatures = await this._runAudioModel();
        const nlpFeatures = await this._runNLPModel(taskText);

        // Fuse and run meta-classifier
        const fusedVector = new Float32Array([
            ...visionFeatures, ...audioFeatures, ...nlpFeatures
        ]);

        const metaTensor = new ort.Tensor('float32', fusedVector, [1, 11]);
        const metaResult = await this.models.meta.run({ input: metaTensor });

        const label = metaResult.label ? metaResult.label.data[0] : 0;
        const probs = metaResult.probabilities ? Array.from(metaResult.probabilities.data) : [0.2, 0.2, 0.2, 0.2, 0.2];

        return {
            flowState: label,
            flowStateName: this.flowStateNames[label],
            flowLabel: this.flowLabels[label],
            flowEmoji: this.flowEmojis[label],
            workQuality: probs[3] + probs[4],
            probabilities: probs,
            confidence: Math.max(...probs),
            timestamp: Date.now(),
        };
    }

    async _runVisionModel() {
        const imageData = this.visionPreprocessor.captureAndPreprocess();
        if (!imageData || !this.models.vision) return new Float32Array(4).fill(0.5);
        const tensor = new ort.Tensor('float32', imageData, [1, 3, 640, 640]);
        const result = await this.models.vision.run({ images: tensor });
        return new Float32Array(4); // Process results
    }

    async _runAudioModel() {
        if (!this.audioExtractor.isRecording || !this.models.audio) return new Float32Array(4).fill(0.5);
        const features = this.audioExtractor.extractFeatures();
        const tensor = new ort.Tensor('float32', features, [1, 52]);
        const result = await this.models.audio.run({ input: tensor });
        return new Float32Array(4); // Process results
    }

    async _runNLPModel(taskText) {
        if (!taskText || !this.models.nlp) return new Float32Array(3).fill(0.5);
        const { inputIds, attentionMask } = this.nlpTokenizer.tokenize(taskText);
        const idsTensor = new ort.Tensor('int64', inputIds, [1, 128]);
        const maskTensor = new ort.Tensor('int64', attentionMask, [1, 128]);
        const result = await this.models.nlp.run({ input_ids: idsTensor, attention_mask: maskTensor });
        return new Float32Array(3); // Process results
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
