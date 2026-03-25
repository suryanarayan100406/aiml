/**
 * ANI Flow Data Collector — Popup Controller
 */
(function () {
    let selectedScore = null;

    // ─── Load Stats ───────────────────────────────────────────
    async function loadStats() {
        const summary = await DataLogger.getSummary();
        document.getElementById('stat-total').textContent = summary.total || 0;
        document.getElementById('stat-tabs').textContent = summary.avgTabs || '—';
        document.getElementById('stat-reports').textContent = summary.selfReports || 0;
        document.getElementById('stat-focus').textContent = summary.avgFocus ? summary.avgFocus.toFixed(1) : '—';
    }

    // ─── Focus Score Buttons ──────────────────────────────────
    document.querySelectorAll('.focus-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.focus-btn').forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
            selectedScore = parseInt(btn.dataset.score);
        });
    });

    // ─── Submit Report ────────────────────────────────────────
    document.getElementById('btn-submit').addEventListener('click', async () => {
        if (!selectedScore) {
            alert('Please select a focus score (1-5)');
            return;
        }
        const task = document.getElementById('task-input').value || '';
        await DataLogger.saveSelfReport(selectedScore, task);

        const toast = document.getElementById('toast');
        toast.style.display = 'block';
        setTimeout(() => { toast.style.display = 'none'; }, 2000);

        selectedScore = null;
        document.querySelectorAll('.focus-btn').forEach(b => b.classList.remove('selected'));
        document.getElementById('task-input').value = '';
        loadStats();
    });

    // ─── Export ───────────────────────────────────────────────
    document.getElementById('btn-export').addEventListener('click', () => {
        DataLogger.downloadCSV();
    });

    // ─── Clear ────────────────────────────────────────────────
    document.getElementById('btn-clear').addEventListener('click', async () => {
        if (confirm('Clear all collected data?')) {
            await DataLogger.clearAll();
            loadStats();
        }
    });

    // ─── Listen for new data ──────────────────────────────────
    chrome.runtime.onMessage.addListener((msg) => {
        if (msg.type === 'DATA_COLLECTED') loadStats();
    });

    // Init
    loadStats();
})();
