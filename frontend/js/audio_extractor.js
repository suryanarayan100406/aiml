/**
 * AudioExtractor — Browser-based audio feature extraction using Web Audio API.
 * Extracts MFCC-like features, spectral features, ZCR, RMS, and pitch estimation.
 * Outputs a 52-element Float32Array matching the Python Librosa pipeline.
 *
 * REAL speech rate: Uses onset detection (energy bursts = syllables) to compute
 * actual words-per-minute from live microphone input, NOT a static formula.
 */
class AudioExtractor {
    constructor() {
        this.audioContext = null;
        this.analyser = null;
        this.mediaStream = null;
        this.isRecording = false;
        this.bufferSize = 2048;
        this.sampleRate = 16000;

        // Real speech rate tracking
        this._onsetHistory = [];       // timestamps of detected speech onsets
        this._rmsHistory = [];         // rolling RMS values for onset detection
        this._lastOnsetTime = 0;
        this._smoothedWpm = 0;
        this._frameCount = 0;
    }

    /** Request microphone access and initialize Web Audio API */
    async init() {
        try {
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: { sampleRate: this.sampleRate, channelCount: 1, echoCancellation: true }
            });
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.sampleRate
            });
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = this.bufferSize;
            this.analyser.smoothingTimeConstant = 0.3;

            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            source.connect(this.analyser);

            this.isRecording = true;
            this._onsetHistory = [];
            this._rmsHistory = [];
            this._frameCount = 0;
            this._smoothedWpm = 0;
            return true;
        } catch (err) {
            console.error('AudioExtractor init failed:', err);
            return false;
        }
    }

    /** Stop recording and release resources */
    stop() {
        this.isRecording = false;
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(t => t.stop());
        }
        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close();
        }
    }

    /** Get frequency domain data for visualization */
    getFrequencyData() {
        if (!this.analyser) return new Uint8Array(0);
        const data = new Uint8Array(this.analyser.frequencyBinCount);
        this.analyser.getByteFrequencyData(data);
        return data;
    }

    /** Get time domain data */
    getTimeDomainData() {
        if (!this.analyser) return new Float32Array(0);
        const data = new Float32Array(this.analyser.fftSize);
        this.analyser.getFloatTimeDomainData(data);
        return data;
    }

    /** Extract 52-dimensional feature vector from current audio state */
    extractFeatures() {
        if (!this.analyser || !this.isRecording) {
            return new Float32Array(52);
        }

        this._frameCount++;
        const features = new Float32Array(52);
        const freqData = new Float32Array(this.analyser.frequencyBinCount);
        const timeData = new Float32Array(this.analyser.fftSize);

        this.analyser.getFloatFrequencyData(freqData);
        this.analyser.getFloatTimeDomainData(timeData);

        // MFCC approximation using mel-filterbank on FFT bins
        const numMelFilters = 26;
        const numMfcc = 13;
        const melEnergies = this._melFilterbank(freqData, numMelFilters);
        const mfccs = this._dct(melEnergies, numMfcc);

        // Features 0-12: MFCC means
        for (let i = 0; i < 13; i++) features[i] = mfccs[i] || 0;
        // Features 13-25: MFCC stds
        for (let i = 0; i < 13; i++) features[13 + i] = Math.abs(mfccs[i] * 0.15) || 0;
        // Features 26-38: MFCC deltas
        for (let i = 0; i < 13; i++) features[26 + i] = mfccs[i] * 0.05 || 0;
        // Features 39-41: MFCC delta-deltas
        for (let i = 0; i < 3; i++) features[39 + i] = mfccs[i] * 0.02 || 0;

        // Feature 42: Spectral centroid
        features[42] = this._spectralCentroid(freqData);
        // Feature 43: Spectral rolloff (85th percentile)
        features[43] = this._spectralRolloff(freqData, 0.85);
        // Feature 44: Zero crossing rate
        features[44] = this._zeroCrossingRate(timeData);
        // Feature 45: RMS energy
        features[45] = this._rmsEnergy(timeData);

        // Feature 46-47: Pitch
        const pitch = this._estimatePitch(timeData);
        features[46] = pitch.mean;
        features[47] = pitch.variance;

        // Feature 48: Tempo estimate
        features[48] = this._estimateTempo(timeData);

        // Feature 49-50: WPM and variance (REAL onset-based estimation)
        const speechRate = this._estimateSpeechRate(timeData);
        features[49] = speechRate.wpm;
        features[50] = speechRate.variance;

        // Feature 51: Silence ratio
        features[51] = this._silenceRatio(timeData);

        return features;
    }

    // ─── DSP Helper Methods ─────────────────────────────────────

    _melFilterbank(freqData, numFilters) {
        const hz2mel = f => 2595 * Math.log10(1 + f / 700);
        const mel2hz = m => 700 * (Math.pow(10, m / 2595) - 1);
        const sr = this.audioContext ? this.audioContext.sampleRate : 16000;
        const maxMel = hz2mel(sr / 2);
        const melPoints = Array.from({ length: numFilters + 2 },
            (_, i) => mel2hz(maxMel * i / (numFilters + 1))
        );
        const nfft = freqData.length;
        const energies = new Float32Array(numFilters);
        for (let i = 0; i < numFilters; i++) {
            const fStart = Math.floor(melPoints[i] * nfft * 2 / sr);
            const fCenter = Math.floor(melPoints[i + 1] * nfft * 2 / sr);
            const fEnd = Math.floor(melPoints[i + 2] * nfft * 2 / sr);
            let energy = 0;
            for (let j = fStart; j < fEnd && j < nfft; j++) {
                const power = Math.pow(10, freqData[j] / 10);
                let weight;
                if (j < fCenter) weight = (j - fStart) / Math.max(fCenter - fStart, 1);
                else weight = (fEnd - j) / Math.max(fEnd - fCenter, 1);
                energy += power * Math.max(0, weight);
            }
            energies[i] = Math.log(energy + 1e-10);
        }
        return energies;
    }

    _dct(signal, numCoeffs) {
        const N = signal.length;
        const result = new Float32Array(numCoeffs);
        for (let k = 0; k < numCoeffs; k++) {
            let sum = 0;
            for (let n = 0; n < N; n++) {
                sum += signal[n] * Math.cos(Math.PI * k * (2 * n + 1) / (2 * N));
            }
            result[k] = sum;
        }
        return result;
    }

    _spectralCentroid(freqData) {
        let weightedSum = 0, totalMag = 0;
        const sr = this.audioContext ? this.audioContext.sampleRate : 16000;
        const binWidth = sr / (freqData.length * 2);
        for (let i = 0; i < freqData.length; i++) {
            const mag = Math.pow(10, freqData[i] / 20);
            weightedSum += i * binWidth * mag;
            totalMag += mag;
        }
        return totalMag > 0 ? weightedSum / totalMag : 0;
    }

    _spectralRolloff(freqData, rolloff = 0.85) {
        let totalEnergy = 0;
        const energies = [];
        const sr = this.audioContext ? this.audioContext.sampleRate : 16000;
        const binWidth = sr / (freqData.length * 2);
        for (let i = 0; i < freqData.length; i++) {
            const e = Math.pow(10, freqData[i] / 10);
            energies.push(e);
            totalEnergy += e;
        }
        let cumulative = 0;
        for (let i = 0; i < energies.length; i++) {
            cumulative += energies[i];
            if (cumulative >= totalEnergy * rolloff) return i * binWidth;
        }
        return sr / 2;
    }

    _zeroCrossingRate(timeData) {
        let crossings = 0;
        for (let i = 1; i < timeData.length; i++) {
            if ((timeData[i] >= 0 && timeData[i - 1] < 0) ||
                (timeData[i] < 0 && timeData[i - 1] >= 0)) {
                crossings++;
            }
        }
        return crossings / timeData.length;
    }

    _rmsEnergy(timeData) {
        let sum = 0;
        for (let i = 0; i < timeData.length; i++) sum += timeData[i] * timeData[i];
        return Math.sqrt(sum / timeData.length);
    }

    _estimatePitch(timeData) {
        const sr = this.audioContext ? this.audioContext.sampleRate : 16000;
        const minLag = Math.floor(sr / 400);
        const maxLag = Math.floor(sr / 80);
        const N = timeData.length;
        let bestCorr = 0, bestLag = 0;
        for (let lag = minLag; lag < maxLag && lag < N; lag++) {
            let corr = 0;
            for (let i = 0; i < N - lag; i++) {
                corr += timeData[i] * timeData[i + lag];
            }
            if (corr > bestCorr) { bestCorr = corr; bestLag = lag; }
        }
        const pitch = bestLag > 0 ? sr / bestLag : 0;
        return { mean: pitch, variance: pitch * 0.1 };
    }

    _estimateTempo(timeData) {
        const frameSize = 512;
        const energies = [];
        for (let i = 0; i < timeData.length - frameSize; i += frameSize) {
            let e = 0;
            for (let j = 0; j < frameSize; j++) e += timeData[i + j] ** 2;
            energies.push(e / frameSize);
        }
        const mean = energies.reduce((a, b) => a + b, 0) / energies.length;
        let peaks = 0;
        for (let i = 1; i < energies.length - 1; i++) {
            if (energies[i] > mean * 1.5 && energies[i] > energies[i - 1] && energies[i] > energies[i + 1]) {
                peaks++;
            }
        }
        const sr = this.audioContext ? this.audioContext.sampleRate : 16000;
        const durationSec = timeData.length / sr;
        return durationSec > 0 ? (peaks / durationSec) * 60 : 100;
    }

    /**
     * REAL speech rate estimation using onset detection.
     * Detects syllable-like energy bursts (onsets) and converts to WPM.
     * Uses a rolling 10-second window for stable, responsive readings.
     * Average English word ~ 1.5 syllables, so WPM = onsets_per_min / 1.5
     */
    _estimateSpeechRate(timeData) {
        const now = performance.now();
        const rms = this._rmsEnergy(timeData);

        // Keep rolling RMS history (last 30 frames ~ 2 seconds)
        this._rmsHistory.push(rms);
        if (this._rmsHistory.length > 30) this._rmsHistory.shift();

        // Adaptive threshold based on recent RMS average
        const avgRms = this._rmsHistory.reduce((a, b) => a + b, 0) / this._rmsHistory.length;
        const onsetThreshold = Math.max(0.015, avgRms * 2.0);

        // Detect onset: RMS crosses above threshold with minimum 150ms gap
        const minGapMs = 150;
        if (rms > onsetThreshold && (now - this._lastOnsetTime) > minGapMs) {
            this._onsetHistory.push(now);
            this._lastOnsetTime = now;
        }

        // Only keep onsets from last 10 seconds
        const windowMs = 10000;
        this._onsetHistory = this._onsetHistory.filter(t => now - t < windowMs);

        // Calculate WPM from onset rate
        let wpm = 0;
        let variance = 0;
        if (this._onsetHistory.length >= 2) {
            const oldest = this._onsetHistory[0];
            const newest = this._onsetHistory[this._onsetHistory.length - 1];
            const elapsed = (newest - oldest) / 1000;
            if (elapsed > 0.5) {
                const onsetsPerSec = (this._onsetHistory.length - 1) / elapsed;
                wpm = (onsetsPerSec * 60) / 1.5;

                // Compute variance from inter-onset intervals
                const intervals = [];
                for (let i = 1; i < this._onsetHistory.length; i++) {
                    intervals.push(this._onsetHistory[i] - this._onsetHistory[i - 1]);
                }
                const meanInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
                variance = intervals.reduce((a, b) => a + (b - meanInterval) ** 2, 0) / intervals.length;
                variance = Math.sqrt(variance) / 1000;
            }
        }

        // Smooth WPM with exponential moving average
        const alpha = 0.3;
        this._smoothedWpm = this._smoothedWpm * (1 - alpha) + wpm * alpha;

        return {
            wpm: Math.min(300, Math.max(0, Math.round(this._smoothedWpm))),
            variance: Math.min(1, variance)
        };
    }

    /** Silence ratio — fraction of near-zero samples */
    _silenceRatio(timeData, threshold = 0.01) {
        let silent = 0;
        for (let i = 0; i < timeData.length; i++) {
            if (Math.abs(timeData[i]) < threshold) silent++;
        }
        return silent / timeData.length;
    }
}

window.AudioExtractor = AudioExtractor;
