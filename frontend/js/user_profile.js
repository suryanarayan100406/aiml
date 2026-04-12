/**
 * UserFlowProfile — Per-user baseline calibration & consistency tracking.
 * Stores data in IndexedDB for persistence across sessions.
 */
class UserFlowProfile {
    constructor(userId = null) {
        // Automatically load last active user or default
        this.userId = userId || localStorage.getItem('ani_active_user') || 'default';
        this.tabBaseline = null;
        this.wpmBaseline = null;
        this.calibrationSessions = 0;
        this.sessionHistory = [];
        this.dbName = 'ANI_FlowDB';
        this.dbVersion = 1;
    }

    /** Open IndexedDB */
    async openDB() {
        return new Promise((resolve, reject) => {
            const req = indexedDB.open(this.dbName, this.dbVersion);
            req.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains('profiles')) {
                    db.createObjectStore('profiles', { keyPath: 'userId' });
                }
                if (!db.objectStoreNames.contains('sessions')) {
                    db.createObjectStore('sessions', { keyPath: 'id', autoIncrement: true });
                }
            };
            req.onsuccess = () => resolve(req.result);
            req.onerror = () => reject(req.error);
        });
    }

    /** Save profile to IndexedDB */
    async save() {
        // Persist the active user
        localStorage.setItem('ani_active_user', this.userId);
        
        const db = await this.openDB();
        const tx = db.transaction('profiles', 'readwrite');
        tx.objectStore('profiles').put({
            userId: this.userId,
            tabBaseline: this.tabBaseline,
            wpmBaseline: this.wpmBaseline,
            calibrationSessions: this.calibrationSessions,
            sessionHistory: this.sessionHistory.slice(-90), // Keep 90 days
        });
        return new Promise((resolve) => { tx.oncomplete = resolve; });
    }

    /** Load profile from IndexedDB */
    async load() {
        const db = await this.openDB();
        const tx = db.transaction('profiles', 'readonly');
        const req = tx.objectStore('profiles').get(this.userId);
        return new Promise((resolve) => {
            req.onsuccess = () => {
                if (req.result) {
                    this.tabBaseline = req.result.tabBaseline;
                    this.wpmBaseline = req.result.wpmBaseline;
                    this.calibrationSessions = req.result.calibrationSessions || 0;
                    this.sessionHistory = req.result.sessionHistory || [];
                }
                resolve(this);
            };
            req.onerror = () => resolve(this);
        });
    }

    /** Calibrate baselines from session data (minimum 5 sessions) */
    calibrate(sessionsData) {
        if (sessionsData.length >= 5) {
            const tabs = sessionsData.map(s => s.tabs).sort((a, b) => a - b);
            this.tabBaseline = tabs[Math.floor(tabs.length / 2)]; // Median
            this.wpmBaseline = sessionsData.reduce((sum, s) => sum + s.wpm, 0) / sessionsData.length;
            this.calibrationSessions = sessionsData.length;
        }
    }

    /** Normalize features relative to this user's baseline */
    normalizeForUser(features) {
        const normalized = { ...features };
        if (this.calibrationSessions >= 5) {
            if (this.tabBaseline && this.tabBaseline > 0) {
                normalized.tab_count_norm = (features.tab_count || 10) / this.tabBaseline;
            }
            if (this.wpmBaseline && this.wpmBaseline > 0) {
                normalized.wpm_norm = (features.wpm || 130) / this.wpmBaseline;
            }
        }
        return normalized;
    }

    /** Add a session result */
    addSession(session) {
        session.timestamp = Date.now();
        session.id = `${this.userId}_${Date.now()}`;
        this.sessionHistory.push(session);
        if (this.sessionHistory.length > 90) {
            this.sessionHistory = this.sessionHistory.slice(-90);
        }
    }

    /** Classify consistency over the last N days */
    classifyConsistency(days = 14) {
        const cutoff = Date.now() - (days * 24 * 60 * 60 * 1000);
        const recent = this.sessionHistory.filter(s => (s.timestamp || 0) > cutoff);

        if (recent.length < 7) return 'INSUFFICIENT_DATA';

        const scores = recent.map(s => s.workQuality || 0.5);
        const n = scores.length;

        // Linear regression slope
        const xMean = (n - 1) / 2;
        const yMean = scores.reduce((a, b) => a + b, 0) / n;
        let num = 0, den = 0;
        for (let i = 0; i < n; i++) {
            num += (i - xMean) * (scores[i] - yMean);
            den += (i - xMean) ** 2;
        }
        const slope = den !== 0 ? num / den : 0;
        const variance = scores.reduce((s, v) => s + (v - yMean) ** 2, 0) / n;

        if (slope > 0.01 && variance < 0.05) return 'IMPROVING';
        if (Math.abs(slope) < 0.005 && variance < 0.05) return 'STABLE';
        if (slope < -0.01) return 'DECLINING';
        return 'INCONSISTENT';
    }

    /** Save a session to IndexedDB sessions store */
    async saveSession(session) {
        const db = await this.openDB();
        const tx = db.transaction('sessions', 'readwrite');
        tx.objectStore('sessions').add(session);
        return new Promise(r => { tx.oncomplete = r; });
    }

    /** Get all sessions from IndexedDB */
    async getAllSessions() {
        const db = await this.openDB();
        const tx = db.transaction('sessions', 'readonly');
        const req = tx.objectStore('sessions').getAll();
        return new Promise(r => {
            req.onsuccess = () => r(req.result || []);
            req.onerror = () => r([]);
        });
    }

    /** Clear all data */
    async clearAll() {
        const db = await this.openDB();
        const tx = db.transaction(['profiles', 'sessions'], 'readwrite');
        tx.objectStore('profiles').clear();
        tx.objectStore('sessions').clear();
        this.tabBaseline = null;
        this.wpmBaseline = null;
        this.calibrationSessions = 0;
        this.sessionHistory = [];
        return new Promise(r => { tx.oncomplete = r; });
    }

    /** Export sessions to CSV string */
    async exportCSV() {
        const sessions = await this.getAllSessions();
        if (sessions.length === 0) return '';
        const headers = Object.keys(sessions[0]).join(',');
        const rows = sessions.map(s =>
            Object.values(s).map(v =>
                typeof v === 'string' ? `"${v}"` : v
            ).join(',')
        );
        return [headers, ...rows].join('\n');
    }
}

// Global instance
window.userProfile = new UserFlowProfile();
