/**
 * ANI Tab Analyzer — Popup Controller
 * Live auto-updating dashboard. No manual input needed.
 */
(function () {
    const $ = id => document.getElementById(id);

    // ─── Update UI with analysis data ─────────────────────────
    function updateUI(data) {
        if (!data) return;

        // Tab count
        $('tab-count-label').textContent = data.tabCount || 0;

        // Productivity score
        const score = Math.round((data.productivityScore || 0) * 100);
        const scoreEl = $('productivity-value');
        const barEl = $('productivity-bar');
        scoreEl.textContent = `${score}%`;

        // Color coding
        let level = 'low';
        if (score >= 60) level = 'high';
        else if (score >= 30) level = 'medium';

        scoreEl.className = `meter-value ${level}`;
        barEl.className = `meter-fill ${level}`;
        barEl.style.width = `${score}%`;

        // Category counts
        const cats = data.categories || {};
        $('count-productive').textContent = cats.productive || 0;
        $('count-distraction').textContent = (cats.distraction || 0) + (cats.news || 0);
        $('count-communication').textContent = cats.communication || 0;
        $('count-neutral').textContent = cats.neutral || 0;

        // Active tab
        if (data.activeTab) {
            $('active-title').textContent = data.activeTab.title || 'Unknown';
            $('active-domain').textContent = data.activeTab.domain || '';
            const badge = $('active-badge');
            const cat = data.activeTab.category || 'neutral';
            badge.textContent = cat.charAt(0).toUpperCase() + cat.slice(1);
            badge.className = `category-badge badge-${cat}`;
        }

        // Unique domains
        $('unique-domains').textContent = `${data.uniqueDomains || 0} domains`;

        // Switch rate
        $('switch-rate').textContent = data.switchRate || 0;

        // Tab list
        const tabList = $('tab-list');
        if (data.tabs && data.tabs.length > 0) {
            tabList.innerHTML = data.tabs.map(tab => `
                <div class="tab-item ${tab.active ? 'active-row' : ''}">
                    <div class="tab-dot ${tab.category}"></div>
                    <span class="tab-title-text" title="${escapeHtml(tab.title)}">${escapeHtml(tab.title)}</span>
                    <span class="tab-domain-text">${tab.domain}</span>
                </div>
            `).join('');
        } else {
            tabList.innerHTML = '<div style="color:#6b7280;font-size:0.7rem;padding:8px;">No tabs found</div>';
        }

        // Connection indicator
        $('conn-dot').classList.remove('disconnected');
    }

    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str || '';
        return div.innerHTML;
    }

    // ─── Fetch data on load and periodically ──────────────────
    async function fetchAndUpdate() {
        try {
            const data = await new Promise((resolve, reject) => {
                chrome.runtime.sendMessage({ type: 'GET_ANALYSIS' }, (response) => {
                    if (chrome.runtime.lastError) {
                        reject(chrome.runtime.lastError);
                    } else {
                        resolve(response);
                    }
                });
            });
            updateUI(data);
        } catch (err) {
            console.warn('[ANI Popup] Fetch error:', err);
            $('conn-dot').classList.add('disconnected');
        }
    }

    // ─── Listen for real-time updates from background ─────────
    chrome.runtime.onMessage.addListener((msg) => {
        if (msg.type === 'ANALYSIS_UPDATE') {
            updateUI(msg.data);
        }
    });

    // ─── Clear button ─────────────────────────────────────────
    $('btn-clear').addEventListener('click', async () => {
        if (confirm('Clear all stored history?')) {
            await new Promise(resolve => {
                chrome.runtime.sendMessage({ type: 'CLEAR_DATA' }, resolve);
            });
        }
    });

    // ─── Init ─────────────────────────────────────────────────
    fetchAndUpdate();
    setInterval(fetchAndUpdate, 5000); // Refresh every 5 seconds
})();
