/**
 * ScreenClassifier — MobileNetV3-Small ONNX inference for screen productivity.
 * Classifies screen captures into 5 categories:
 *   PRODUCTIVE_CODE | PRODUCTIVE_DOCS | COMMUNICATION | DISTRACTION | NEUTRAL
 *
 * Input: 224×224 RGB image (ImageNet-normalized, NCHW format)
 * Output: { screenClass, className, confidence, productivityScore }
 */
class ScreenClassifier {
    constructor() {
        this.session = null;
        this.loaded = false;
        this.inputSize = 224;

        // Must match training script (screen_class_mapping.json)
        this.classNames = [
            'PRODUCTIVE_CODE',
            'PRODUCTIVE_DOCS',
            'COMMUNICATION',
            'DISTRACTION',
            'NEUTRAL',
        ];

        this.productivityScores = {
            0: 0.95,  // Code → very productive
            1: 0.80,  // Docs → productive
            2: 0.50,  // Communication → neutral
            3: 0.10,  // Distraction → not productive
            4: 0.40,  // Neutral → slightly below average
        };

        // ImageNet normalization (must match training)
        this.mean = [0.485, 0.456, 0.406];
        this.std  = [0.229, 0.224, 0.225];

        // Internal canvas for resizing
        this.canvas = document.createElement('canvas');
        this.canvas.width = this.inputSize;
        this.canvas.height = this.inputSize;
        this.ctx = this.canvas.getContext('2d');

        // Smoothing: keep last N predictions for stability
        this.historySize = 5;
        this.predictionHistory = [];
    }

    /**
     * Load the ONNX model.
     * @returns {boolean} true if loaded successfully
     */
    async load(modelPath = '../models/screen_classifier.onnx') {
        try {
            if (typeof ort === 'undefined') {
                console.warn('[ScreenClassifier] ONNX Runtime not available');
                return false;
            }

            // Try loading the class mapping first (for custom configs)
            try {
                const mappingResp = await fetch('../models/screen_class_mapping.json');
                if (mappingResp.ok) {
                    const mapping = await mappingResp.json();
                    if (mapping.classes) {
                        this.classNames = Object.values(mapping.classes);
                    }
                    if (mapping.productivity_scores) {
                        this.productivityScores = {};
                        for (const [k, v] of Object.entries(mapping.productivity_scores)) {
                            this.productivityScores[parseInt(k)] = v;
                        }
                    }
                    if (mapping.normalization) {
                        this.mean = mapping.normalization.mean;
                        this.std = mapping.normalization.std;
                    }
                    if (mapping.input_size) {
                        this.inputSize = mapping.input_size;
                        this.canvas.width = this.inputSize;
                        this.canvas.height = this.inputSize;
                    }
                    console.log('[ScreenClassifier] Loaded class mapping');
                }
            } catch (e) {
                console.warn('[ScreenClassifier] No class mapping found, using defaults');
            }

            this.session = await ort.InferenceSession.create(modelPath);
            this.loaded = true;
            console.log('✅ ScreenClassifier loaded (MobileNetV3-Small)');
            return true;
        } catch (e) {
            console.warn('⚠️ ScreenClassifier model not found:', e.message);
            this.loaded = false;
            return false;
        }
    }

    /**
     * Classify a screen capture frame.
     * @param {HTMLVideoElement} videoElement - The screen capture video element
     * @returns {Object} Classification result
     */
    async classify(videoElement) {
        if (!this.loaded || !this.session) {
            return this._fallbackClassify(videoElement);
        }

        try {
            const tensor = this._preprocessFrame(videoElement);
            const inputName = this.session.inputNames[0] || 'input';
            const feeds = {};
            feeds[inputName] = new ort.Tensor('float32', tensor, [1, 3, this.inputSize, this.inputSize]);

            const result = await this.session.run(feeds);
            const outputName = this.session.outputNames[0] || 'output';
            const logits = Array.from(result[outputName].data);

            // Softmax
            const maxLogit = Math.max(...logits);
            const expLogits = logits.map(l => Math.exp(l - maxLogit));
            const sumExp = expLogits.reduce((a, b) => a + b, 0);
            const probs = expLogits.map(e => e / sumExp);

            const predictedClass = probs.indexOf(Math.max(...probs));
            const confidence = Math.max(...probs);

            // Add to history for smoothing
            this.predictionHistory.push({ cls: predictedClass, conf: confidence, probs });
            if (this.predictionHistory.length > this.historySize) {
                this.predictionHistory.shift();
            }

            // Smoothed prediction (majority vote weighted by confidence)
            const smoothed = this._getSmoothedPrediction();

            return {
                screenClass: smoothed.cls,
                className: this.classNames[smoothed.cls] || 'UNKNOWN',
                confidence: smoothed.conf,
                productivityScore: this.productivityScores[smoothed.cls] ?? 0.5,
                rawProbs: probs,
                rawClass: predictedClass,
                source: 'ONNX Model',
            };
        } catch (e) {
            console.warn('[ScreenClassifier] Inference failed:', e);
            return this._fallbackClassify(videoElement);
        }
    }

    /**
     * Preprocess video frame for MobileNetV3 (224×224, ImageNet normalized, NCHW).
     */
    _preprocessFrame(videoElement) {
        const vw = videoElement.videoWidth;
        const vh = videoElement.videoHeight;

        // Resize to 224×224 (center crop + resize)
        const scale = Math.max(this.inputSize / vw, this.inputSize / vh);
        const sw = Math.round(vw * scale);
        const sh = Math.round(vh * scale);
        const ox = Math.round((this.inputSize - sw) / 2);
        const oy = Math.round((this.inputSize - sh) / 2);

        this.ctx.fillStyle = '#808080';
        this.ctx.fillRect(0, 0, this.inputSize, this.inputSize);
        this.ctx.drawImage(videoElement, ox, oy, sw, sh);

        const imageData = this.ctx.getImageData(0, 0, this.inputSize, this.inputSize);
        const { data } = imageData;
        const pixels = this.inputSize * this.inputSize;

        // NCHW format with ImageNet normalization
        const tensor = new Float32Array(3 * pixels);
        for (let i = 0; i < pixels; i++) {
            const srcIdx = i * 4;
            tensor[i]              = (data[srcIdx]     / 255.0 - this.mean[0]) / this.std[0]; // R
            tensor[pixels + i]     = (data[srcIdx + 1] / 255.0 - this.mean[1]) / this.std[1]; // G
            tensor[2 * pixels + i] = (data[srcIdx + 2] / 255.0 - this.mean[2]) / this.std[2]; // B
        }

        return tensor;
    }

    /**
     * Get smoothed prediction from history (majority vote weighted by confidence).
     */
    _getSmoothedPrediction() {
        if (this.predictionHistory.length === 0) {
            return { cls: 4, conf: 0.5 };
        }

        // Weighted vote
        const classWeights = new Float32Array(this.classNames.length);
        for (const pred of this.predictionHistory) {
            classWeights[pred.cls] += pred.conf;
        }

        const bestClass = classWeights.indexOf(Math.max(...classWeights));
        const avgConf = classWeights[bestClass] / this.predictionHistory.length;

        return { cls: bestClass, conf: Math.min(0.99, avgConf) };
    }

    /**
     * Fallback classification using basic image analysis (no ONNX model).
     * Analyzes color distributions and layout patterns.
     */
    _fallbackClassify(videoElement) {
        if (!videoElement || !videoElement.videoWidth) {
            return {
                screenClass: 4,
                className: 'NEUTRAL',
                confidence: 0.3,
                productivityScore: 0.5,
                rawProbs: [0.2, 0.2, 0.2, 0.2, 0.2],
                source: 'no-screen',
            };
        }

        // Draw frame and analyze
        this.ctx.drawImage(videoElement, 0, 0, this.inputSize, this.inputSize);
        const imageData = this.ctx.getImageData(0, 0, this.inputSize, this.inputSize);
        const { data } = imageData;
        const pixels = this.inputSize * this.inputSize;

        // Compute basic image statistics
        let totalR = 0, totalG = 0, totalB = 0;
        let darkPixels = 0;
        let brightPixels = 0;
        let colorfulPixels = 0;

        for (let i = 0; i < pixels; i++) {
            const r = data[i * 4];
            const g = data[i * 4 + 1];
            const b = data[i * 4 + 2];
            totalR += r;
            totalG += g;
            totalB += b;

            const brightness = (r + g + b) / 3;
            if (brightness < 60) darkPixels++;
            if (brightness > 200) brightPixels++;

            // Colorfulness: high saturation
            const max = Math.max(r, g, b);
            const min = Math.min(r, g, b);
            if (max - min > 80) colorfulPixels++;
        }

        const avgR = totalR / pixels;
        const avgG = totalG / pixels;
        const avgB = totalB / pixels;
        const darkRatio = darkPixels / pixels;
        const brightRatio = brightPixels / pixels;
        const colorfulRatio = colorfulPixels / pixels;
        const avgBrightness = (avgR + avgG + avgB) / 3;

        // Heuristic classification
        const scores = new Float32Array(5);

        // PRODUCTIVE_CODE: dark background, low brightness, code-like patterns
        scores[0] = darkRatio * 0.5 + (1 - avgBrightness / 255) * 0.3 + (1 - colorfulRatio) * 0.2;

        // PRODUCTIVE_DOCS: bright background, low colorfulness
        scores[1] = brightRatio * 0.4 + (avgBrightness / 255) * 0.3 + (1 - colorfulRatio) * 0.3;

        // COMMUNICATION: mixed dark/bright, moderate colorfulness
        scores[2] = (1 - Math.abs(darkRatio - 0.3)) * 0.3 + (1 - Math.abs(colorfulRatio - 0.15)) * 0.3 + 0.2;

        // DISTRACTION: highly colorful, varied
        scores[3] = colorfulRatio * 0.5 + (Math.abs(avgR - avgG) + Math.abs(avgG - avgB)) / 255 * 0.3 + 0.1;

        // NEUTRAL: moderate everything
        scores[4] = (1 - Math.abs(avgBrightness / 255 - 0.5)) * 0.4 + (1 - colorfulRatio) * 0.3 + 0.2;

        // Normalize
        const total = scores.reduce((a, b) => a + b, 0);
        const probs = Array.from(scores).map(s => s / total);
        const predictedClass = probs.indexOf(Math.max(...probs));

        return {
            screenClass: predictedClass,
            className: this.classNames[predictedClass],
            confidence: Math.max(...probs),
            productivityScore: this.productivityScores[predictedClass] ?? 0.5,
            rawProbs: probs,
            source: 'heuristic',
        };
    }

    /** Reset prediction history */
    reset() {
        this.predictionHistory = [];
    }
}

window.ScreenClassifier = ScreenClassifier;
