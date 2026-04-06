/**
 * ANI Flow Data Collector — Background Service Worker
 * Counts open tabs every 30 minutes and logs data for meta-classifier training.
 */

const ALARM_NAME = 'ani_data_collection';
const COLLECTION_INTERVAL_MINUTES = 30;

// ─── Alarm Setup ──────────────────────────────────────────────
chrome.runtime.onInstalled.addListener(() => {
    chrome.alarms.create(ALARM_NAME, {
        delayInMinutes: 1,
        periodInMinutes: COLLECTION_INTERVAL_MINUTES,
    });
    console.log(`[ANI] Data collection alarm set: every ${COLLECTION_INTERVAL_MINUTES} minutes`);
});

// ─── Alarm Handler ────────────────────────────────────────────
chrome.alarms.onAlarm.addListener(async (alarm) => {
    if (alarm.name !== ALARM_NAME) return;

    try {
        const data = await collectData();
        await storeDataPoint(data);

        // Notify popup if open
        chrome.runtime.sendMessage({
            type: 'DATA_COLLECTED',
            data: data,
        }).catch(() => {}); // Popup may not be open

        console.log('[ANI] Data point collected:', data);
    } catch (err) {
        console.error('[ANI] Collection error:', err);
    }
});

// ─── Data Collection ──────────────────────────────────────────
async function collectData() {
    // Count open tabs
    const tabs = await chrome.tabs.query({});
    const tabCount = tabs.length;

    // Get active tab info
    const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const activeTitle = activeTab?.title || 'Unknown';
    const activeUrl = activeTab?.url || '';

    // Classify the active tab domain
    const domain = activeUrl ? new URL(activeUrl).hostname : '';
    const tabCategory = classifyDomain(domain);

    // Compute tab diversity (unique domains)
    const domains = new Set(tabs.map(t => {
        try { return new URL(t.url).hostname; } catch { return ''; }
    }).filter(Boolean));

    return {
        timestamp: new Date().toISOString(),
        tabCount: tabCount,
        uniqueDomains: domains.size,
        activeTitle: activeTitle.substring(0, 100),
        activeDomain: domain,
        tabCategory: tabCategory,
        tabCountNorm: Math.min(tabCount / 30, 1.0),
    };
}

// ─── Domain Classification ────────────────────────────────────
function classifyDomain(domain) {
    const categories = {
        work: ['github.com', 'gitlab.com', 'stackoverflow.com', 'docs.google.com',
               'notion.so', 'figma.com', 'linear.app', 'jira.atlassian.net',
               'slack.com', 'teams.microsoft.com', 'vscode.dev'],
        social: ['twitter.com', 'x.com', 'facebook.com', 'instagram.com',
                 'reddit.com', 'tiktok.com', 'youtube.com', 'twitch.tv'],
        news: ['news.ycombinator.com', 'bbc.com', 'cnn.com', 'nytimes.com',
               'techcrunch.com', 'theverge.com'],
        shopping: ['amazon.com', 'ebay.com', 'etsy.com'],
        email: ['mail.google.com', 'outlook.com', 'yahoo.com'],
    };

    for (const [category, domains] of Object.entries(categories)) {
        if (domains.some(d => domain.includes(d))) return category;
    }
    return 'other';
}

// ─── Storage ──────────────────────────────────────────────────
async function storeDataPoint(data) {
    const result = await chrome.storage.local.get('dataPoints');
    const points = result.dataPoints || [];
    points.push(data);

    // Keep last 2000 data points (~40 days at 30-min intervals)
    if (points.length > 2000) points.splice(0, points.length - 2000);

    await chrome.storage.local.set({ dataPoints: points });
}

// ─── Message Handler ──────────────────────────────────────────
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.type === 'GET_TAB_COUNT') {
        chrome.tabs.query({}).then(tabs => {
            sendResponse({ tabCount: tabs.length });
        });
        return true; // Async response
    }

    if (msg.type === 'GET_TAB_DATA') {
        getFullTabData().then(data => {
            sendResponse(data);
        });
        return true;
    }

    if (msg.type === 'GET_ALL_DATA') {
        chrome.storage.local.get('dataPoints').then(result => {
            sendResponse({ data: result.dataPoints || [] });
        });
        return true;
    }

    if (msg.type === 'SAVE_SELF_REPORT') {
        storeDataPoint({
            ...msg.data,
            timestamp: new Date().toISOString(),
            type: 'self_report',
        }).then(() => sendResponse({ ok: true }));
        return true;
    }

    if (msg.type === 'CLEAR_DATA') {
        chrome.storage.local.set({ dataPoints: [] }).then(() => {
            sendResponse({ ok: true });
        });
        return true;
    }
});

// ─── Full Tab Data for Frontend ───────────────────────────────
async function getFullTabData() {
    const tabs = await chrome.tabs.query({});
    const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });

    // Categorize all tabs
    let workTabs = 0, socialTabs = 0, entertainmentTabs = 0, otherTabs = 0;
    const domains = new Set();

    tabs.forEach(tab => {
        try {
            const domain = new URL(tab.url).hostname;
            domains.add(domain);
            const cat = classifyDomain(domain);
            if (cat === 'work' || cat === 'email') workTabs++;
            else if (cat === 'social') socialTabs++;
            else if (cat === 'news' || cat === 'shopping') entertainmentTabs++;
            else otherTabs++;
        } catch { otherTabs++; }
    });

    return {
        tabCount: tabs.length,
        uniqueDomains: domains.size,
        workTabs,
        socialTabs,
        entertainmentTabs,
        otherTabs,
        activeTitle: activeTab?.title || 'Unknown',
        activeDomain: activeTab?.url ? new URL(activeTab.url).hostname : '',
        activeCategory: activeTab?.url ? classifyDomain(new URL(activeTab.url).hostname) : 'other',
        distractionTabs: socialTabs + entertainmentTabs,
        tabCountNorm: Math.min(tabs.length / 30, 1.0),
        timestamp: Date.now(),
        extensionConnected: true,
    };
}

// ─── Proactive Tab Change Push ────────────────────────────────
// Push tab updates to all ANI dashboard tabs when tabs change
async function pushTabUpdate() {
    const data = await getFullTabData();
    // Send to all tabs — content script will filter
    const tabs = await chrome.tabs.query({});
    for (const tab of tabs) {
        try {
            chrome.tabs.sendMessage(tab.id, { type: 'TAB_DATA_UPDATE', data }).catch(() => {});
        } catch {}
    }
}

// Listen for tab events
chrome.tabs.onCreated.addListener(pushTabUpdate);
chrome.tabs.onRemoved.addListener(pushTabUpdate);
chrome.tabs.onActivated.addListener(pushTabUpdate);

