/**
 * NLPTokenizer — Lightweight DistilBERT WordPiece tokenizer in pure JS.
 * Handles tokenization, padding, and truncation for browser-side ONNX inference.
 */
class NLPTokenizer {
    constructor() {
        this.vocab = null;
        this.vocabReverse = null;
        this.maxLength = 128;
        this.padTokenId = 0;
        this.unkTokenId = 100;
        this.clsTokenId = 101;
        this.sepTokenId = 102;
        this.loaded = false;

        // Task class names
        this.classNames = ['DEEP_WORK', 'SHALLOW_WORK', 'CREATIVE', 'ADMINISTRATIVE', 'COMMUNICATION'];
        this.demandMap = { 0: 0.9, 1: 0.2, 2: 0.7, 3: 0.3, 4: 0.5 };
    }

    /** Load vocabulary file */
    async loadVocab(vocabUrl = '../models/vocab.txt') {
        try {
            const response = await fetch(vocabUrl);
            if (!response.ok) {
                console.warn('Vocab file not found, using fallback tokenization');
                return false;
            }
            const text = await response.text();
            const tokens = text.split('\n').filter(t => t.length > 0);
            this.vocab = new Map();
            this.vocabReverse = new Map();
            tokens.forEach((token, idx) => {
                this.vocab.set(token, idx);
                this.vocabReverse.set(idx, token);
            });
            this.loaded = true;
            return true;
        } catch {
            console.warn('Failed to load vocab, using fallback');
            return false;
        }
    }

    /** Tokenize text to input_ids and attention_mask */
    tokenize(text) {
        if (this.loaded) {
            return this._wordPieceTokenize(text);
        }
        return this._fallbackTokenize(text);
    }

    /** Full WordPiece tokenization */
    _wordPieceTokenize(text) {
        const tokens = [this.clsTokenId]; // [CLS]

        // Basic pre-tokenization: lowercase, split on whitespace and punctuation
        const words = text.toLowerCase()
            .replace(/[^\w\s'-]/g, ' $& ')
            .split(/\s+/)
            .filter(w => w.length > 0);

        for (const word of words) {
            if (tokens.length >= this.maxLength - 1) break;

            // Try to find the whole word
            if (this.vocab.has(word)) {
                tokens.push(this.vocab.get(word));
                continue;
            }

            // WordPiece: try to split into subwords
            let remaining = word;
            let isFirst = true;

            while (remaining.length > 0 && tokens.length < this.maxLength - 1) {
                let found = false;
                for (let end = remaining.length; end > 0; end--) {
                    const sub = isFirst ? remaining.slice(0, end) : '##' + remaining.slice(0, end);
                    if (this.vocab.has(sub)) {
                        tokens.push(this.vocab.get(sub));
                        remaining = remaining.slice(end);
                        isFirst = false;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    tokens.push(this.unkTokenId); // [UNK]
                    break;
                }
            }
        }

        tokens.push(this.sepTokenId); // [SEP]

        // Pad or truncate to maxLength
        const inputIds = new BigInt64Array(this.maxLength);
        const attentionMask = new BigInt64Array(this.maxLength);

        for (let i = 0; i < this.maxLength; i++) {
            if (i < tokens.length) {
                inputIds[i] = BigInt(tokens[i]);
                attentionMask[i] = 1n;
            } else {
                inputIds[i] = BigInt(this.padTokenId);
                attentionMask[i] = 0n;
            }
        }

        return { inputIds, attentionMask, tokenCount: tokens.length };
    }

    /** Fallback tokenization when vocab is not available */
    _fallbackTokenize(text) {
        const words = text.toLowerCase().split(/\s+/).filter(w => w.length > 0);
        const inputIds = new BigInt64Array(this.maxLength);
        const attentionMask = new BigInt64Array(this.maxLength);

        inputIds[0] = BigInt(this.clsTokenId);
        attentionMask[0] = 1n;

        for (let i = 0; i < words.length && i + 1 < this.maxLength - 1; i++) {
            // Simple hash-based token ID 
            let hash = 0;
            for (let j = 0; j < words[i].length; j++) {
                hash = ((hash << 5) - hash) + words[i].charCodeAt(j);
                hash = hash & 0x7FFF; // Keep in vocab range
            }
            inputIds[i + 1] = BigInt(Math.max(1000, hash % 30000));
            attentionMask[i + 1] = 1n;
        }

        const endIdx = Math.min(words.length + 1, this.maxLength - 1);
        inputIds[endIdx] = BigInt(this.sepTokenId);
        attentionMask[endIdx] = 1n;

        return { inputIds, attentionMask, tokenCount: endIdx + 1 };
    }

    /** Demo-mode NLP classification (keyword-based when no ONNX model) */
    classifyDemo(text) {
        const lower = text.toLowerCase();

        // Keyword-based classification
        const scores = new Float32Array(5);

        // DEEP_WORK keywords
        const deepWords = ['implement', 'debug', 'refactor', 'optimize', 'algorithm', 'architecture',
            'database', 'pipeline', 'migrate', 'build', 'design system', 'compiler', 'ml', 'model'];
        // SHALLOW_WORK keywords
        const shallowWords = ['update', 'fix typo', 'rename', 'add logging', 'bump version',
            'clean up', 'format', 'lint', 'config', 'env', 'readme'];
        // CREATIVE keywords
        const creativeWords = ['design', 'brainstorm', 'prototype', 'wireframe', 'visual',
            'illustration', 'animation', 'color', 'typography', 'ux'];
        // ADMINISTRATIVE keywords
        const adminWords = ['review', 'approve', 'schedule', 'meeting', 'report', 'audit',
            'organize', 'plan', 'budget', 'onboarding'];
        // COMMUNICATION keywords
        const commWords = ['email', 'present', 'write blog', 'reply', 'draft', 'proposal',
            'newsletter', 'demo', 'tutorial', 'communicate'];

        const wordSets = [deepWords, shallowWords, creativeWords, adminWords, commWords];

        wordSets.forEach((words, idx) => {
            words.forEach(w => {
                if (lower.includes(w)) scores[idx] += 1.5;
            });
        });

        // Add small random noise
        for (let i = 0; i < 5; i++) scores[i] += Math.random() * 0.5;

        // Softmax
        const maxScore = Math.max(...scores);
        const expScores = Array.from(scores).map(s => Math.exp(s - maxScore));
        const sumExp = expScores.reduce((a, b) => a + b, 0);
        const probs = expScores.map(e => e / sumExp);

        const predictedClass = probs.indexOf(Math.max(...probs));

        return {
            taskClass: predictedClass,
            className: this.classNames[predictedClass],
            cognitiveDemand: this.demandMap[predictedClass],
            confidence: Math.max(...probs),
            probabilities: probs,
        };
    }
}

window.NLPTokenizer = NLPTokenizer;
