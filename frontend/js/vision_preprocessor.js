/**
 * VisionPreprocessor — Screenshot preprocessing for YOLOv8 inference.
 * Handles screen capture, resize to 640×640, RGB normalization, NCHW tensor conversion.
 */
class VisionPreprocessor {
    constructor() {
        this.targetSize = 640;
        this.canvas = document.createElement('canvas');
        this.canvas.width = this.targetSize;
        this.canvas.height = this.targetSize;
        this.ctx = this.canvas.getContext('2d');
        this.mediaStream = null;
        this.video = null;
    }

    /** Initialize screen capture */
    async initScreenCapture() {
        try {
            this.mediaStream = await navigator.mediaDevices.getDisplayMedia({
                video: { width: 1920, height: 1080, frameRate: 1 }
            });
            this.video = document.createElement('video');
            this.video.srcObject = this.mediaStream;
            this.video.play();
            return true;
        } catch (err) {
            console.error('Screen capture failed:', err);
            return false;
        }
    }

    /** Stop screen capture */
    stop() {
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(t => t.stop());
        }
    }

    /** Capture current frame and preprocess for YOLO */
    captureAndPreprocess() {
        if (!this.video || !this.video.videoWidth) return null;

        // Draw video frame to canvas at target size
        this.ctx.drawImage(this.video, 0, 0, this.targetSize, this.targetSize);
        const imageData = this.ctx.getImageData(0, 0, this.targetSize, this.targetSize);

        return this.preprocessImage(imageData);
    }

    /** Convert ImageData to NCHW Float32Array tensor normalized to [0, 1] */
    preprocessImage(imageData) {
        const { data, width, height } = imageData;
        const pixels = width * height;

        // NCHW format: [1, 3, 640, 640]
        const tensor = new Float32Array(3 * pixels);

        for (let i = 0; i < pixels; i++) {
            const srcIdx = i * 4;
            tensor[i] = data[srcIdx] / 255.0;                  // R channel
            tensor[pixels + i] = data[srcIdx + 1] / 255.0;      // G channel
            tensor[2 * pixels + i] = data[srcIdx + 2] / 255.0;  // B channel
        }

        return tensor;
    }

    /** Extract vision features without ONNX model (demo mode) */
    extractDemoFeatures() {
        if (!this.video || !this.video.videoWidth) {
            return {
                tab_count_norm: Math.random() * 0.6 + 0.1,
                phone_visible: Math.random() > 0.7 ? 1 : 0,
                distraction_count_norm: Math.random() * 0.4,
                focus_ratio: Math.random() * 0.5 + 0.3,
            };
        }

        // Analyze actual captured image
        this.ctx.drawImage(this.video, 0, 0, this.targetSize, this.targetSize);
        const imageData = this.ctx.getImageData(0, 0, this.targetSize, this.targetSize);

        return this._analyzeScreenshot(imageData);
    }

    /** Basic image analysis for demo mode */
    _analyzeScreenshot(imageData) {
        const { data, width, height } = imageData;

        // Analyze top portion for tab bar (darker = more tabs typically)
        const tabBarHeight = 40;
        let tabBarBrightness = 0;
        let tabBarVariance = 0;
        const tabPixels = width * tabBarHeight;
        const brightnesses = [];

        for (let y = 0; y < tabBarHeight; y++) {
            for (let x = 0; x < width; x++) {
                const idx = (y * width + x) * 4;
                const b = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
                tabBarBrightness += b;
                brightnesses.push(b);
            }
        }
        tabBarBrightness /= tabPixels;
        const mean = tabBarBrightness;
        for (const b of brightnesses) {
            tabBarVariance += (b - mean) ** 2;
        }
        tabBarVariance /= brightnesses.length;

        // High variance in tab bar = many tabs (tab separators create edges)
        const tabCountNorm = Math.min(1, tabBarVariance / 2000);

        // Overall brightness of bottom half — lower = fewer work tools
        let bottomBrightness = 0;
        const bottomStart = Math.floor(height * 0.5);
        const bottomPixels = width * (height - bottomStart);
        for (let y = bottomStart; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = (y * width + x) * 4;
                bottomBrightness += (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
            }
        }
        bottomBrightness /= bottomPixels;

        // Focus ratio — if main content area is uniform, likely focused on one thing
        const focusRatio = 1 - Math.min(1, tabBarVariance / 3000);

        return {
            tab_count_norm: tabCountNorm,
            phone_visible: 0, // Can't reliably detect without ML
            distraction_count_norm: Math.random() * 0.3,
            focus_ratio: Math.max(0.1, Math.min(1.0, focusRatio)),
        };
    }
}

window.VisionPreprocessor = VisionPreprocessor;
