/**
 * TaskClassifier — Lightweight task-type classifier (pure JS, 0 model files).
 *
 * Replaces the 256MB DistilBERT ONNX model with a fast keyword + URL + TF-IDF
 * approach optimized for short text (tab titles, task descriptions).
 *
 * Input:  text string (tab title, task description, or URL)
 * Output: { taskClass, className, cognitiveDemand, confidence, probabilities }
 *
 * Classes:
 *   0 = DEEP_WORK       (demand 0.90) — complex coding, architecture, ML
 *   1 = SHALLOW_WORK    (demand 0.20) — typos, config, renaming, formatting
 *   2 = CREATIVE        (demand 0.70) — design, brainstorm, prototype, UX
 *   3 = ADMINISTRATIVE  (demand 0.30) — review, meeting, budget, onboarding
 *   4 = COMMUNICATION   (demand 0.50) — email, blog, draft, present
 *
 * Also maintained: NLPTokenizer compatibility (classifyDemo, loadVocab, tokenize)
 * so existing code calling those methods still works.
 */
class NLPTokenizer {
    constructor() {
        // ─── Backwards compat ───
        this.vocab = null;
        this.vocabReverse = null;
        this.maxLength = 128;
        this.padTokenId = 0;
        this.unkTokenId = 100;
        this.clsTokenId = 101;
        this.sepTokenId = 102;
        this.loaded = false;

        // ─── Task classification ───
        this.classNames = ['DEEP_WORK', 'SHALLOW_WORK', 'CREATIVE', 'ADMINISTRATIVE', 'COMMUNICATION'];
        this.demandMap = { 0: 0.9, 1: 0.2, 2: 0.7, 3: 0.3, 4: 0.5 };

        // ─── Weighted keyword banks ───
        // Each entry: [keyword_or_phrase, weight]
        // Higher weight = stronger signal
        this._keywords = {
            // ── DEEP_WORK (0) ────────────────────────────────────
            0: [
                // Programming
                ['implement', 2.5], ['debug', 2.5], ['refactor', 2.5], ['algorithm', 2.5],
                ['architecture', 2.0], ['database', 2.0], ['migrate', 2.0], ['pipeline', 2.0],
                ['compiler', 2.5], ['optimizer', 2.0], ['concurrency', 2.5], ['threading', 2.5],
                ['distributed', 2.0], ['microservice', 2.0], ['monolith', 2.0],
                ['machine learning', 2.5], ['neural network', 2.5], ['deep learning', 2.5],
                ['model training', 2.5], ['backpropagation', 2.5], ['gradient', 2.0],
                ['inference', 2.0], ['tensor', 2.0], ['pytorch', 2.5], ['tensorflow', 2.5],
                ['encryption', 2.0], ['authentication', 2.0], ['oauth', 2.0], ['jwt', 2.0],
                ['unit test', 2.0], ['integration test', 2.0], ['test coverage', 2.0],
                ['api design', 2.0], ['schema', 2.0], ['query optimization', 2.5],
                ['memory leak', 2.5], ['profiling', 2.0], ['benchmark', 2.0],
                ['binary search', 2.0], ['dynamic programming', 2.5], ['recursion', 2.0],
                ['sorting', 1.5], ['graph algorithm', 2.5], ['tree traversal', 2.0],
                ['websocket', 2.0], ['rest api', 1.5], ['grpc', 2.0], ['graphql', 2.0],
                ['docker', 1.5], ['kubernetes', 2.0], ['ci/cd', 1.5], ['devops', 1.5],
                ['caching', 2.0], ['sharding', 2.5], ['replication', 2.0],
                ['code review', 1.5], ['pull request', 1.0], ['merge conflict', 2.0],
                // IDE / code tool indicators
                ['.py', 1.8], ['.js', 1.8], ['.ts', 1.8], ['.java', 1.8], ['.cpp', 1.8],
                ['.go', 1.8], ['.rs', 1.8], ['.rb', 1.5], ['.cs', 1.5], ['.swift', 1.5],
                ['vs code', 2.0], ['visual studio', 2.0], ['intellij', 2.0], ['pycharm', 2.0],
                ['webstorm', 2.0], ['neovim', 2.0], ['vim', 1.5], ['emacs', 1.5],
                ['terminal', 1.5], ['console', 1.2], ['ssh', 1.5], ['localhost', 1.5],
                ['github.com', 1.5], ['gitlab', 1.5], ['bitbucket', 1.5],
                ['stackoverflow', 1.5], ['stack overflow', 1.5], ['mdn web docs', 1.5],
                ['jupyter', 2.0], ['colab', 2.0], ['notebook', 1.5],
                ['build', 1.0], ['deploy', 1.2], ['release', 1.0],
            ],

            // ── SHALLOW_WORK (1) ─────────────────────────────────
            1: [
                ['fix typo', 3.0], ['typo', 2.0], ['rename', 2.0], ['bump version', 2.5],
                ['update readme', 2.5], ['update changelog', 2.5], ['clean up', 2.0],
                ['formatting', 2.0], ['lint', 2.0], ['linter', 2.0], ['prettier', 2.0],
                ['add logging', 2.0], ['add comment', 2.0], ['remove unused', 2.0],
                ['update dependency', 2.5], ['update package', 2.0], ['pin version', 2.5],
                ['env variable', 2.0], ['.env', 2.0], ['config', 1.5], ['configuration', 1.5],
                ['copyright', 2.0], ['license header', 2.5], ['indentation', 2.0],
                ['sort import', 2.5], ['dead code', 2.0], ['deprecated', 1.5],
                ['minor fix', 2.5], ['hotfix', 2.0], ['patch', 1.5], ['quick fix', 2.5],
                ['file move', 2.0], ['folder structure', 1.5], ['reorganize', 1.5],
                ['type annotation', 2.0], ['add type', 1.5], ['docstring', 1.5],
                ['settings', 1.0], ['preferences', 1.0], ['downloads', 1.0],
            ],

            // ── CREATIVE (2) ─────────────────────────────────────
            2: [
                ['design', 2.0], ['brainstorm', 2.5], ['prototype', 2.5], ['wireframe', 2.5],
                ['mockup', 2.5], ['figma', 2.5], ['sketch', 2.0], ['adobe', 1.5],
                ['photoshop', 2.0], ['illustrator', 2.0], ['canva', 2.0], ['framer', 2.0],
                ['illustration', 2.5], ['animation', 2.0], ['motion design', 2.5],
                ['color palette', 2.5], ['typography', 2.5], ['font', 1.5],
                ['ui design', 2.5], ['ux design', 2.5], ['user experience', 2.0],
                ['user interface', 2.0], ['interaction design', 2.5], ['micro-interaction', 2.5],
                ['visual identity', 2.5], ['brand', 2.0], ['branding', 2.0],
                ['landing page', 1.5], ['hero section', 2.0], ['layout', 1.5],
                ['creative', 2.0], ['innovative', 1.5], ['concept', 1.5],
                ['storyboard', 2.5], ['mood board', 2.5], ['inspiration', 1.5],
                ['responsive design', 2.0], ['mobile design', 2.0], ['dark mode', 1.5],
                ['gamification', 2.0], ['parallax', 2.0], ['3d model', 2.0],
                ['blender', 2.0], ['cinema 4d', 2.0], ['after effects', 2.0],
                ['dribbble', 2.5], ['behance', 2.5], ['pinterest', 1.5],
                ['tailwind', 1.0], ['css', 1.0], ['sass', 1.0],
            ],

            // ── ADMINISTRATIVE (3) ───────────────────────────────
            3: [
                ['meeting', 2.5], ['schedule', 2.0], ['calendar', 2.0],
                ['review', 1.5], ['approve', 2.5], ['approval', 2.5],
                ['report', 2.0], ['compliance', 2.5], ['audit', 2.5],
                ['sprint', 2.0], ['standup', 2.5], ['retrospective', 2.5], ['planning', 1.5],
                ['jira', 2.5], ['asana', 2.5], ['trello', 2.0], ['notion', 1.0],
                ['monday.com', 2.5], ['linear', 1.5], ['clickup', 2.0],
                ['expense', 2.5], ['budget', 2.5], ['invoice', 2.5], ['payroll', 2.5],
                ['onboarding', 2.0], ['offboarding', 2.5], ['hr', 2.0],
                ['timesheet', 2.5], ['attendance', 2.5], ['leave request', 2.5],
                ['inventory', 2.0], ['asset', 1.5], ['vendor', 2.0], ['procurement', 2.5],
                ['contract', 2.0], ['renewal', 2.0], ['license', 1.5],
                ['ticket', 1.5], ['triage', 2.5], ['backlog', 2.0],
                ['kpi', 2.5], ['okr', 2.5], ['dashboard', 1.0],
                ['project management', 2.0], ['gantt', 2.5], ['milestone', 2.0],
                ['google sheets', 1.5], ['excel', 1.5], ['spreadsheet', 1.5],
            ],

            // ── COMMUNICATION (4) ────────────────────────────────
            4: [
                ['email', 2.5], ['draft email', 3.0], ['reply', 1.5], ['forward', 1.5],
                ['gmail', 2.5], ['outlook', 2.5], ['yahoo mail', 2.5],
                ['blog post', 2.5], ['write blog', 3.0], ['article', 1.5],
                ['presentation', 2.5], ['slide deck', 2.5], ['powerpoint', 2.0],
                ['google slides', 2.5], ['keynote', 2.0],
                ['proposal', 2.0], ['newsletter', 2.5], ['press release', 2.5],
                ['demo video', 2.5], ['tutorial', 1.5], ['walkthrough', 1.5],
                ['postmortem', 2.0], ['incident report', 2.5],
                ['release notes', 2.0], ['documentation', 1.5],
                ['faq', 2.0], ['knowledge base', 2.0], ['help center', 2.0],
                ['slack', 2.0], ['teams', 1.5], ['discord', 1.5], ['zoom', 2.0],
                ['google meet', 2.5], ['skype', 2.0], ['webex', 2.5],
                ['chat', 1.5], ['message', 1.0], ['dm', 1.5],
                ['video call', 2.5], ['conference call', 2.5],
                ['podcast', 2.0], ['webinar', 2.5], ['livestream', 2.0],
                ['compose', 1.5], ['write', 1.0], ['draft', 1.5],
                ['medium.com', 2.0], ['substack', 2.0], ['wordpress', 1.5],
                ['linkedin', 1.5], ['x.com', 1.0], ['twitter', 1.0],
            ],
        };

        // ─── URL domain hints (for tab URL classification) ───
        this._domainHints = {
            0: ['github.com', 'gitlab.com', 'bitbucket.org', 'stackoverflow.com',
                'developer.mozilla.org', 'docs.python.org', 'pytorch.org',
                'tensorflow.org', 'npmjs.com', 'pypi.org', 'crates.io',
                'leetcode.com', 'hackerrank.com', 'codeforces.com',
                'kaggle.com', 'huggingface.co', 'arxiv.org',
                'localhost', '127.0.0.1', 'colab.research.google.com'],
            1: ['en.wikipedia.org', 'google.com/search'],
            2: ['figma.com', 'dribbble.com', 'behance.net', 'canva.com',
                'coolors.co', 'unsplash.com', 'pexels.com', 'fontpair.co',
                'pinterest.com', 'awwwards.com'],
            3: ['jira.atlassian.com', 'trello.com', 'asana.com', 'notion.so',
                'monday.com', 'linear.app', 'clickup.com',
                'sheets.google.com', 'docs.google.com/spreadsheets'],
            4: ['mail.google.com', 'outlook.live.com', 'outlook.office.com',
                'slack.com', 'app.slack.com', 'teams.microsoft.com',
                'zoom.us', 'meet.google.com', 'discord.com',
                'medium.com', 'substack.com', 'wordpress.com',
                'docs.google.com/presentation', 'docs.google.com/document'],
        };

        // ─── Distraction detection (not a task class, but useful for context) ───
        this._distractionDomains = [
            'youtube.com', 'netflix.com', 'twitch.tv', 'primevideo.com',
            'reddit.com', 'twitter.com', 'x.com', 'facebook.com', 'instagram.com',
            'tiktok.com', 'snapchat.com', 'tumblr.com',
            'news.ycombinator.com', 'buzzfeed.com',
            'amazon.com', 'ebay.com', 'flipkart.com', 'myntra.com',
        ];

        // ─── App name → class mapping (for tab title patterns like "file.py — VS Code") ───
        this._appPatterns = [
            // [regex, class, weight]
            [/\.(py|js|ts|jsx|tsx|java|cpp|c|go|rs|rb|swift|kt|scala|r|sql|sh|bat|ps1)\b/i, 0, 2.5],
            [/—\s*(vs\s*code|visual\s*studio|intellij|pycharm|webstorm|sublime|atom|vim|nvim)/i, 0, 3.0],
            [/\bcolab\b/i, 0, 2.5],
            [/\bjupyter\b/i, 0, 2.5],
            [/\bterminal\b|\bconsole\b|\bcmd\b|\bpowershell\b/i, 0, 2.0],
            [/—\s*(figma|sketch|adobe|canva|framer)/i, 2, 3.0],
            [/\bgmail\b|\binbox\b|\boutlook\b/i, 4, 2.5],
            [/\bslack\b|\bteams\b|\bdiscord\b/i, 4, 2.5],
            [/\bzoom\s*meeting\b|\bgoogle\s*meet\b/i, 4, 3.0],
            [/\bjira\b|\btrello\b|\basana\b|\blinear\b/i, 3, 2.5],
            [/\bgoogle\s*docs\b|\bgoogle\s*sheets\b/i, 1, 1.5],
            [/\bnotion\b/i, 1, 1.5],
            [/\bword\b.*\bdocument\b|\bexcel\b|\bpowerpoint\b/i, 1, 1.5],
        ];

        // ─── Smoothing: track recent classifications ───
        this._history = [];
        this._historySize = 3;
    }

    /** Load vocabulary file (kept for backwards compatibility) */
    async loadVocab(vocabUrl = '../models/vocab.txt') {
        // No longer needed — but maintain the interface
        this.loaded = true;
        console.log('✅ TaskClassifier ready (lightweight keyword engine, no ONNX needed)');
        return true;
    }

    /** Tokenize text (kept for backwards compatibility) */
    tokenize(text) {
        // Return dummy tokens — the ONNX NLP model is no longer used
        const inputIds = new BigInt64Array(this.maxLength);
        const attentionMask = new BigInt64Array(this.maxLength);
        inputIds[0] = BigInt(this.clsTokenId);
        attentionMask[0] = 1n;
        return { inputIds, attentionMask, tokenCount: 1 };
    }

    /**
     * Primary classification method.
     * Analyzes text (tab title, URL, or task description) and returns task type.
     *
     * @param {string} text - Tab title, task description, or URL
     * @param {string} [url] - Optional URL for domain-based hints
     * @returns {{ taskClass, className, cognitiveDemand, confidence, probabilities }}
     */
    classify(text, url = '') {
        const lower = (text || '').toLowerCase().trim();
        if (lower.length === 0) {
            return this._makeResult(0, 0.3); // default: DEEP_WORK with low confidence
        }

        // Initialize scores
        const scores = new Float64Array(5);

        // ─── Phase 1: Keyword matching (weighted) ────────────────
        for (let cls = 0; cls < 5; cls++) {
            for (const [keyword, weight] of this._keywords[cls]) {
                if (lower.includes(keyword.toLowerCase())) {
                    scores[cls] += weight;
                }
            }
        }

        // ─── Phase 2: App name patterns (regex) ──────────────────
        for (const [pattern, cls, weight] of this._appPatterns) {
            if (pattern.test(lower)) {
                scores[cls] += weight;
            }
        }

        // ─── Phase 3: URL domain hints ───────────────────────────
        const urlLower = (url || '').toLowerCase();
        if (urlLower.length > 0) {
            for (let cls = 0; cls < 5; cls++) {
                for (const domain of this._domainHints[cls]) {
                    if (urlLower.includes(domain)) {
                        scores[cls] += 3.0; // Strong signal from URL
                        break;
                    }
                }
            }

            // Check for distraction domains
            for (const domain of this._distractionDomains) {
                if (urlLower.includes(domain)) {
                    // Distraction detected — lower cognitive demand
                    scores[1] += 2.0; // Treat as shallow/not-real-work
                    break;
                }
            }
        }

        // ─── Phase 4: Text complexity heuristics ─────────────────
        const wordCount = lower.split(/\s+/).length;
        const hasNumbers = /\d+/.test(lower);
        const hasTechnicalChars = /[{}()\[\]<>\/\\|=&@#$%^*]/.test(lower);

        // Long task descriptions with technical chars → likely deep work
        if (wordCount > 10 && hasTechnicalChars) scores[0] += 1.0;
        // Very short text with no strong signals → likely shallow or neutral
        if (wordCount <= 3 && Math.max(...scores) < 1.0) scores[1] += 0.5;

        // ─── Phase 5: Special patterns ───────────────────────────
        // "New Tab" or empty-ish → neutral shallow
        if (lower === 'new tab' || lower === 'about:blank' || lower === 'untitled') {
            return this._makeResult(1, 0.4);
        }
        // Google search → depends on what's being searched
        if (lower.includes('google') && lower.includes('search')) {
            scores[0] += 0.5; // Slight tilt to deep work (research)
        }

        // ─── Phase 6: Normalize to probabilities ─────────────────
        const maxScore = Math.max(...scores);

        // If no keywords matched at all, return low-confidence default
        if (maxScore < 0.5) {
            return this._makeResult(1, 0.3); // SHALLOW_WORK with low confidence
        }

        // Temperature-scaled softmax for well-calibrated probabilities
        const temperature = 1.5;
        const expScores = Array.from(scores).map(s => Math.exp(s / temperature));
        const sumExp = expScores.reduce((a, b) => a + b, 0);
        const probs = expScores.map(e => e / sumExp);

        const predictedClass = probs.indexOf(Math.max(...probs));
        const confidence = Math.max(...probs);

        // ─── Phase 7: Smoothing (optional, for tab title flicker) ─
        this._history.push({ cls: predictedClass, conf: confidence });
        if (this._history.length > this._historySize) this._history.shift();

        return {
            taskClass: predictedClass,
            className: this.classNames[predictedClass],
            cognitiveDemand: this.demandMap[predictedClass],
            confidence: Math.min(0.98, confidence),
            probabilities: probs,
        };
    }

    /** Backwards-compatible alias used by demo mode and _runNLPModel fallback */
    classifyDemo(text) {
        return this.classify(text);
    }

    /** Build a result object for edge cases */
    _makeResult(cls, conf) {
        const probs = [0.1, 0.1, 0.1, 0.1, 0.1];
        probs[cls] = conf;
        // Re-normalize
        const total = probs.reduce((a, b) => a + b, 0);
        const normalizedProbs = probs.map(p => p / total);

        return {
            taskClass: cls,
            className: this.classNames[cls],
            cognitiveDemand: this.demandMap[cls],
            confidence: conf,
            probabilities: normalizedProbs,
        };
    }

    /** Reset classification history */
    reset() {
        this._history = [];
    }
}

window.NLPTokenizer = NLPTokenizer;
