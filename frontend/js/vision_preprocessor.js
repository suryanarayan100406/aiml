/**
 * VisionPreprocessor — Webcam + Screenshot preprocessing for YOLOv8 inference.
 * Supports two modes:
 *   1. Webcam capture — detects phone, distractions on desk via camera
 *   2. Screen capture — analyzes screenshots for tab bars
 *
 * Handles resize to 640×640, RGB normalization, NCHW tensor conversion.
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
        this.mode = null; // 'webcam' | 'screen' | null

        // Live preview element (shown in UI)
        this.previewCanvas = null;
        this.previewCtx = null;

        // COCO class mapping for pretrained model
        // Maps COCO class index → our desk category
        this.cocoToDeskMap = {
            67: { id: 0, name: 'phone', type: 'distraction' },
            63: { id: 1, name: 'laptop', type: 'workspace' },
            62: { id: 1, name: 'monitor', type: 'workspace' },
            66: { id: 2, name: 'keyboard', type: 'work_tool' },
            64: { id: 2, name: 'mouse', type: 'work_tool' },
            65: { id: 3, name: 'remote', type: 'distraction' },
            73: { id: 3, name: 'book', type: 'neutral' },
            41: { id: 3, name: 'cup', type: 'neutral' },
            39: { id: 3, name: 'bottle', type: 'neutral' },
        };

        // Detection results (updated each frame)
        this.lastDetections = [];
    }

    /** Initialize webcam capture */
    async initWebcam() {
        try {
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'environment' // Prefer back camera on mobile
                }
            });
            this.video = document.createElement('video');
            this.video.srcObject = this.mediaStream;
            this.video.setAttribute('playsinline', '');
            await this.video.play();
            this.mode = 'webcam';

            console.log(`[Vision] Webcam initialized: ${this.video.videoWidth}×${this.video.videoHeight}`);
            return true;
        } catch (err) {
            console.error('Webcam capture failed:', err);
            return false;
        }
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
            this.mode = 'screen';
            return true;
        } catch (err) {
            console.error('Screen capture failed:', err);
            return false;
        }
    }

    /** Stop capture */
    stop() {
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(t => t.stop());
        }
        this.mode = null;
    }

    /** Check if webcam/screen is active */
    get isActive() { return this.mode !== null && this.video && this.video.videoWidth > 0; }

    /** Capture current frame and preprocess for YOLO */
    captureAndPreprocess() {
        if (!this.video || !this.video.videoWidth) return null;

        // Draw video frame to canvas at target size (letterboxed)
        const vw = this.video.videoWidth;
        const vh = this.video.videoHeight;
        const scale = Math.min(this.targetSize / vw, this.targetSize / vh);
        const sw = Math.round(vw * scale);
        const sh = Math.round(vh * scale);
        const ox = Math.round((this.targetSize - sw) / 2);
        const oy = Math.round((this.targetSize - sh) / 2);

        // Clear canvas (letterbox padding = gray)
        this.ctx.fillStyle = '#808080';
        this.ctx.fillRect(0, 0, this.targetSize, this.targetSize);
        this.ctx.drawImage(this.video, ox, oy, sw, sh);

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

    /**
     * Parse YOLO detections from ONNX output and map to desk categories.
     * Works with both pretrained COCO model (80 classes) and fine-tuned model (4 classes).
     * @param {Object} output - ONNX model output tensor
     * @param {boolean} isCoco - Whether the model outputs 80 COCO classes
     * @returns {Object} Vision features for the meta-classifier
     */
    parseDetections(output, isCoco = true) {
        const data = output.data;
        const dims = output.dims;

        // YOLOv8 output format: [1, 84, 8400] for COCO (80 classes + 4 box coords)
        // Or [1, 8, 8400] for our fine-tuned model (4 classes + 4 box coords)
        let numBoxes, numClasses;

        if (dims.length === 3) {
            // [batch, features, num_boxes] — YOLOv8 native format
            numBoxes = dims[2];
            numClasses = dims[1] - 4; // subtract 4 box coords
        } else if (dims.length === 2) {
            // [num_detections, 6] — post-processed format (x1,y1,x2,y2,conf,class)
            return this._parsePostProcessed(data, dims[0], isCoco);
        } else {
            console.warn('[Vision] Unexpected output dims:', dims);
            return this._defaultFeatures();
        }

        const confThreshold = 0.25;
        let phoneVisible = 0;
        let monitorCount = 0;
        let workToolCount = 0;
        let distractionCount = 0;
        const detections = [];

        for (let i = 0; i < numBoxes; i++) {
            // YOLOv8 raw output: each column is a detection
            // First 4 rows: cx, cy, w, h
            // Remaining rows: class confidences
            const cx = data[0 * numBoxes + i];
            const cy = data[1 * numBoxes + i];
            const w = data[2 * numBoxes + i];
            const h = data[3 * numBoxes + i];

            // Find max class confidence
            let maxConf = 0;
            let maxClass = 0;
            for (let c = 0; c < numClasses; c++) {
                const conf = data[(4 + c) * numBoxes + i];
                if (conf > maxConf) {
                    maxConf = conf;
                    maxClass = c;
                }
            }

            if (maxConf < confThreshold) continue;

            // Map to desk category
            let deskCategory = null;
            if (isCoco) {
                deskCategory = this.cocoToDeskMap[maxClass];
            } else {
                // Fine-tuned: classes are already our 4 categories
                const names = ['phone', 'monitor', 'work_tool', 'distraction'];
                deskCategory = { id: maxClass, name: names[maxClass] || 'unknown' };
            }

            if (!deskCategory) continue;

            detections.push({
                box: [cx - w/2, cy - h/2, cx + w/2, cy + h/2], // xyxy
                confidence: maxConf,
                classId: deskCategory.id,
                className: deskCategory.name,
            });

            // Accumulate counts
            switch (deskCategory.id) {
                case 0: phoneVisible = 1; break;
                case 1: monitorCount++; break;
                case 2: workToolCount++; break;
                case 3: distractionCount++; break;
            }
        }

        this.lastDetections = detections;

        return {
            tab_count_norm: Math.min(monitorCount / 3, 1.0), // monitors as proxy for tabs
            phone_visible: phoneVisible,
            distraction_count_norm: Math.min(distractionCount / 5, 1.0),
            focus_ratio: monitorCount > 0 || workToolCount > 0
                ? Math.max(0.3, 1 - (distractionCount / Math.max(1, monitorCount + workToolCount + distractionCount)))
                : 0.5,
            detections: detections,
            source: 'yolo-webcam',
        };
    }

    /** Parse post-processed YOLO output (already NMS'd) */
    _parsePostProcessed(data, numDetections, isCoco) {
        let phoneVisible = 0, monitorCount = 0, workToolCount = 0, distractionCount = 0;
        const detections = [];

        for (let i = 0; i < numDetections; i++) {
            const offset = i * 6;
            const conf = data[offset + 4];
            const cls = Math.round(data[offset + 5]);
            if (conf < 0.3) continue;

            let deskCategory = isCoco ? this.cocoToDeskMap[cls] : { id: cls, name: ['phone','monitor','work_tool','distraction'][cls] };
            if (!deskCategory) continue;

            detections.push({
                box: [data[offset], data[offset+1], data[offset+2], data[offset+3]],
                confidence: conf,
                classId: deskCategory.id,
                className: deskCategory.name,
            });

            switch (deskCategory.id) {
                case 0: phoneVisible = 1; break;
                case 1: monitorCount++; break;
                case 2: workToolCount++; break;
                case 3: distractionCount++; break;
            }
        }

        this.lastDetections = detections;

        return {
            tab_count_norm: Math.min(monitorCount / 3, 1.0),
            phone_visible: phoneVisible,
            distraction_count_norm: Math.min(distractionCount / 5, 1.0),
            focus_ratio: monitorCount > 0 || workToolCount > 0
                ? Math.max(0.3, 1 - (distractionCount / Math.max(1, monitorCount + workToolCount + distractionCount)))
                : 0.5,
            detections: detections,
            source: 'yolo-webcam',
        };
    }

    /** Default features when no model/webcam available */
    _defaultFeatures() {
        return {
            tab_count_norm: 0.3,
            phone_visible: 0,
            distraction_count_norm: 0.2,
            focus_ratio: 0.6,
            detections: [],
            source: 'default',
        };
    }

    /**
     * Draw detection bounding boxes onto a preview canvas.
     * Call this each frame for live visualization.
     */
    drawDetections(targetCanvas) {
        if (!this.video || !this.video.videoWidth || !targetCanvas) return;

        const ctx = targetCanvas.getContext('2d');
        const w = targetCanvas.width;
        const h = targetCanvas.height;

        // Draw video frame
        ctx.drawImage(this.video, 0, 0, w, h);

        // Draw bounding boxes
        const colors = {
            phone: '#EF4444',     // Red — distraction!
            monitor: '#06B6D4',   // Cyan — workspace
            laptop: '#06B6D4',
            work_tool: '#10B981', // Green — working
            keyboard: '#10B981',
            mouse: '#10B981',
            distraction: '#F59E0B', // Amber — distraction
            remote: '#F59E0B',
            book: '#F59E0B',
            cup: '#6B7280',       // Gray — neutral
            bottle: '#6B7280',
        };

        const scaleX = w / this.targetSize;
        const scaleY = h / this.targetSize;

        for (const det of this.lastDetections) {
            const [x1, y1, x2, y2] = det.box;
            const bx = x1 * scaleX;
            const by = y1 * scaleY;
            const bw = (x2 - x1) * scaleX;
            const bh = (y2 - y1) * scaleY;

            const color = colors[det.className] || '#8B5CF6';

            // Box
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(bx, by, bw, bh);

            // Label background
            const label = `${det.className} ${(det.confidence * 100).toFixed(0)}%`;
            ctx.font = 'bold 12px Inter, sans-serif';
            const textW = ctx.measureText(label).width + 8;
            ctx.fillStyle = color;
            ctx.fillRect(bx, by - 20, textW, 20);

            // Label text
            ctx.fillStyle = '#fff';
            ctx.fillText(label, bx + 4, by - 6);
        }

        // Status overlay
        if (this.lastDetections.length === 0) {
            ctx.fillStyle = 'rgba(0,0,0,0.5)';
            ctx.fillRect(0, h - 30, w, 30);
            ctx.fillStyle = '#F59E0B';
            ctx.font = '12px Inter, sans-serif';
            ctx.fillText('No objects detected — point camera at your desk', 10, h - 10);
        }
    }

    /** Extract vision features without ONNX model (demo mode) — uses webcam image analysis */
    extractDemoFeatures() {
        if (!this.video || !this.video.videoWidth) {
            return {
                tab_count_norm: Math.random() * 0.6 + 0.1,
                phone_visible: Math.random() > 0.7 ? 1 : 0,
                distraction_count_norm: Math.random() * 0.4,
                focus_ratio: Math.random() * 0.5 + 0.3,
                detections: [],
                source: 'simulated',
            };
        }

        // Analyze actual captured image (heuristic-based when no YOLO model)
        this.ctx.drawImage(this.video, 0, 0, this.targetSize, this.targetSize);
        const imageData = this.ctx.getImageData(0, 0, this.targetSize, this.targetSize);
        const result = this._analyzeScreenshot(imageData);
        result.source = this.mode === 'webcam' ? 'webcam-heuristic' : 'screen-heuristic';
        return result;
    }

    /** Basic image analysis for demo mode */
    _analyzeScreenshot(imageData) {
        const { data, width, height } = imageData;

        // Analyze top portion for tab bar
        const tabBarHeight = 40;
        let tabBarVariance = 0;
        const brightnesses = [];

        for (let y = 0; y < tabBarHeight; y++) {
            for (let x = 0; x < width; x++) {
                const idx = (y * width + x) * 4;
                brightnesses.push((data[idx] + data[idx + 1] + data[idx + 2]) / 3);
            }
        }

        const mean = brightnesses.reduce((a, b) => a + b, 0) / brightnesses.length;
        for (const b of brightnesses) tabBarVariance += (b - mean) ** 2;
        tabBarVariance /= brightnesses.length;

        const tabCountNorm = Math.min(1, tabBarVariance / 2000);
        const focusRatio = 1 - Math.min(1, tabBarVariance / 3000);

        return {
            tab_count_norm: tabCountNorm,
            phone_visible: 0,
            distraction_count_norm: Math.random() * 0.3,
            focus_ratio: Math.max(0.1, Math.min(1.0, focusRatio)),
            detections: [],
        };
    }
}

window.VisionPreprocessor = VisionPreprocessor;
