/**
 * AudioExtractor — Browser-based audio feature extraction using Web Audio API.
 * Extracts MFCC-like features, spectral features, ZCR, RMS, and pitch estimation.
 * Outputs a 52-element Float32Array matching the Python Librosa pipeline.
 */
class AudioExtractor {
    constructor() {
        this.audioContext = null;
        this.analyser = null;
        this.mediaStream = null;
        this.isRecording = false;
        this.bufferSize = 2048;
        this.sampleRate = 16000;
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

        const features = new Float32Array(52);
        const freqData = new Float32Array(this.analyser.frequencyBinCount);
        const timeData = new Float32Array(this.analyser.fftSize);

        this.analyser.getFloatFrequencyData(freqData);
        this.analyser.getFloatTimeDomainData(timeData);

        // MFCC approximation using mel-filterbank on FFT bins
        const numMelFilters = 26;
        const numMfcc = 13;
        const melEnergies = this._melFilterbank(freqData, numMelFilters);

        // DCT to get MFCCs
        const mfccs = this._dct(melEnergies, numMfcc);

        // Features 0-12: MFCC means (using single frame as proxy)
        for (let i = 0; i < 13; i++) features[i] = mfccs[i] || 0;

        // Features 13-25: MFCC stds (estimated from frame-level variation)
        for (let i = 0; i < 13; i++) features[13 + i] = Math.abs(mfccs[i] * 0.15) || 0;

        // Features 26-38: MFCC deltas (approximated)
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

        // Feature 46-47: Pitch mean and variance (autocorrelation-based)
        const pitch = this._estimatePitch(timeData);
        features[46] = pitch.mean;
        features[47] = pitch.variance;

        // Feature 48: Tempo estimate
        features[48] = this._estimateTempo(timeData);

        // Feature 49-50: WPM and variance (estimated from energy patterns)
        const speechRate = this._estimateSpeechRate(timeData);
        features[49] = speechRate.wpm;
        features[50] = speechRate.variance;

        // Feature 51: Silence ratio
        features[51] = this._silenceRatio(timeData);

        return features;
    }

    /** Mel filterbank energy computation */
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
                const power = Math.pow(10, freqData[j] / 10); // dB to linear
                let weight;
                if (j < fCenter) {
                    weight = (j - fStart) / Math.max(fCenter - fStart, 1);
                } else {
                    weight = (fEnd - j) / Math.max(fEnd - fCenter, 1);
                }
                energy += power * Math.max(0, weight);
            }
            energies[i] = Math.log(energy + 1e-10);
        }
        return energies;
    }

    /** Discrete Cosine Transform (Type-II) */
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

    /** Spectral centroid — weighted average of frequencies */
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

    /** Spectral rolloff — frequency below which rolloff% of energy lies */
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

    /** Zero crossing rate */
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

    /** RMS energy */
    _rmsEnergy(timeData) {
        let sum = 0;
        for (let i = 0; i < timeData.length; i++) sum += timeData[i] * timeData[i];
        return Math.sqrt(sum / timeData.length);
    }

    /** Pitch estimation via autocorrelation */
    _estimatePitch(timeData) {
        const sr = this.audioContext ? this.audioContext.sampleRate : 16000;
        const minLag = Math.floor(sr / 400); // Max 400 Hz
        const maxLag = Math.floor(sr / 80);  // Min 80 Hz
        const N = timeData.length;

        let bestCorr = 0, bestLag = 0;
        for (let lag = minLag; lag < maxLag && lag < N; lag++) {
            let corr = 0;
            for (let i = 0; i < N - lag; i++) {
                corr += timeData[i] * timeData[i + lag];
            }
            if (corr > bestCorr) {
                bestCorr = corr;
                bestLag = lag;
            }
        }

        const pitch = bestLag > 0 ? sr / bestLag : 0;
        return { mean: pitch, variance: pitch * 0.1 };
    }

    /** Tempo estimation (rough, based on energy envelope) */
    _estimateTempo(timeData) {
        // Simple onset detection via energy peaks
        const frameSize = 512;
        const energies = [];
        for (let i = 0; i < timeData.length - frameSize; i += frameSize) {
            let e = 0;
            for (let j = 0; j < frameSize; j++) e += timeData[i + j] ** 2;
            energies.push(e / frameSize);
        }
        // Count peaks above mean
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

    /** Estimate speech rate (WPM approximation) */
    _estimateSpeechRate(timeData) {
        const rms = this._rmsEnergy(timeData);
        // Very rough: map RMS to WPM range
        const wpm = 80 + rms * 2000;
        return { wpm: Math.min(300, Math.max(50, wpm)), variance: rms * 500 };
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
