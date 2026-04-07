/**
 * ANI Flow Data Collector — Background Service Worker
 * Automatically analyzes ALL open Chrome tabs every 5 seconds.
 * Classifies each tab, computes productivity score, and pushes
 * the analysis to the ANI dashboard in real-time.
 *
 * NO manual self-reports. Fully automated.
 */

// ─── Configuration ────────────────────────────────────────────
const PUSH_INTERVAL_MS = 5000; // Push tab analysis every 5 seconds
const SWITCH_WINDOW_MS = 60000; // Track switches in last 60 seconds

// ─── State ────────────────────────────────────────────────────
let lastActiveTabId = null;
let tabSwitches = []; // timestamps of recent tab switches

// ─── Domain Classification Database ──────────────────────────
const DOMAIN_RULES = {
    productive: [
        'github.com', 'gitlab.com', 'bitbucket.org',
        'stackoverflow.com', 'stackexchange.com',
        'docs.google.com', 'sheets.google.com', 'slides.google.com',
        'notion.so', 'figma.com', 'canva.com',
        'linear.app', 'jira.atlassian.net', 'trello.com', 'asana.com',
        'vscode.dev', 'codepen.io', 'codesandbox.io', 'replit.com',
        'kaggle.com', 'colab.research.google.com',
        'aws.amazon.com', 'console.cloud.google.com', 'portal.azure.com',
        'vercel.com', 'netlify.com', 'heroku.com',
        'medium.com', 'dev.to', 'hashnode.dev',
        'arxiv.org', 'scholar.google.com', 'researchgate.net',
        'udemy.com', 'coursera.org', 'edx.org', 'khanacademy.org',
        'w3schools.com', 'mdn.mozilla.org', 'developer.mozilla.org',
        'npmjs.com', 'pypi.org', 'crates.io',
        'localhost', '127.0.0.1',
    ],
    distraction: [
        'youtube.com', 'twitch.tv', 'netflix.com', 'primevideo.com',
        'facebook.com', 'instagram.com', 'tiktok.com', 'snapchat.com',
        'twitter.com', 'x.com', 'threads.net',
        'reddit.com', '9gag.com', 'imgur.com',
        'amazon.com', 'flipkart.com', 'ebay.com', 'etsy.com',
        'myntra.com', 'ajio.com', 'meesho.com',
        'buzzfeed.com', 'boredpanda.com',
        'play.google.com', 'store.steampowered.com',
        'spotify.com', 'music.apple.com', 'soundcloud.com',
        'pinterest.com', 'tumblr.com',
    ],
    communication: [
        'mail.google.com', 'outlook.com', 'outlook.office.com',
        'yahoo.com', 'protonmail.com',
        'slack.com', 'app.slack.com',
        'teams.microsoft.com', 'discord.com', 'discord.gg',
        'web.whatsapp.com', 'web.telegram.org', 'signal.org',
        'zoom.us', 'meet.google.com',
        'chat.google.com', 'messages.google.com',
        'linkedin.com',
    ],
    news: [
        'news.ycombinator.com', 'bbc.com', 'cnn.com', 'nytimes.com',
        'theguardian.com', 'reuters.com', 'apnews.com',
        'techcrunch.com', 'theverge.com', 'wired.com', 'arstechnica.com',
        'ndtv.com', 'timesofindia.indiatimes.com', 'thehindu.com',
        'news.google.com',
    ],
};

// ─── Classify a domain ────────────────────────────────────────
function classifyDomain(domain) {
    if (!domain) return 'neutral';
    const d = domain.toLowerCase();
    for (const [category, domains] of Object.entries(DOMAIN_RULES)) {
        if (domains.some(rule => d.includes(rule))) return category;
    }
    // Heuristic: if domain contains common productive keywords
    if (d.includes('docs') || d.includes('wiki') || d.includes('api') || d.includes('dashboard')) {
        return 'productive';
    }
    return 'neutral';
}

// ─── Classify tab title for extra context ─────────────────────
function classifyTitle(title) {
    if (!title) return null;
    const t = title.toLowerCase();
    // Override domain classification based on title content
    if (t.includes('game') || t.includes('play') || t.includes('watch')) return 'distraction';
    if (t.includes('email') || t.includes('inbox') || t.includes('compose')) return 'communication';
    if (t.includes('code') || t.includes('debug') || t.includes('pull request') || t.includes('commit')) return 'productive';
    return null; // No title-based override
}

// ─── Full Tab Analysis ────────────────────────────────────────
async function analyzeAllTabs() {
    const tabs = await chrome.tabs.query({});
    const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });

    const categories = { productive: 0, distraction: 0, communication: 0, news: 0, neutral: 0 };
    const domains = new Set();
    const tabDetails = [];

    for (const tab of tabs) {
        let domain = '';
        try { domain = new URL(tab.url).hostname; } catch { domain = ''; }
        domains.add(domain);

        // Classify by domain first, then override by title if stronger signal
        let category = classifyDomain(domain);
        const titleCategory = classifyTitle(tab.title);
        if (titleCategory) category = titleCategory;

        categories[category] = (categories[category] || 0) + 1;

        tabDetails.push({
            id: tab.id,
            title: (tab.title || '').substring(0, 120),
            domain: domain,
            category: category,
            active: tab.id === activeTab?.id,
            audible: tab.audible || false,
            pinned: tab.pinned || false,
        });
    }

    // Compute productivity score (0 to 1)
    const total = tabs.length || 1;
    const productivityScore = categories.productive / total;
    const distractionScore = (categories.distraction + categories.news * 0.5) / total;

    // Tab switch rate (switches in last minute)
    const now = Date.now();
    tabSwitches = tabSwitches.filter(ts => now - ts < SWITCH_WINDOW_MS);
    const switchRate = tabSwitches.length;

    // Active tab info
    let activeDomain = '';
    try { activeDomain = new URL(activeTab?.url || '').hostname; } catch {}
    const activeCategory = classifyDomain(activeDomain);

    return {
        tabCount: tabs.length,
        uniqueDomains: domains.size,
        tabs: tabDetails,
        activeTab: {
            title: (activeTab?.title || 'Unknown').substring(0, 120),
            domain: activeDomain,
            category: activeCategory,
            url: activeTab?.url || '',
        },
        categories: categories,
        productivityScore: Math.round(productivityScore * 100) / 100,
        distractionScore: Math.round(distractionScore * 100) / 100,
        switchRate: switchRate,
        tabCountNorm: Math.min(tabs.length / 30, 1.0),
        extensionConnected: true,
        timestamp: Date.now(),
    };
}

// ─── Push Analysis to ANI Dashboard ───────────────────────────
async function pushAnalysis() {
    try {
        const analysis = await analyzeAllTabs();

        // Store data point for history
        await storeDataPoint(analysis);

        // Send to ALL tabs — content script will forward to ANI frontend
        const tabs = await chrome.tabs.query({});
        for (const tab of tabs) {
            try {
                chrome.tabs.sendMessage(tab.id, {
                    type: 'TAB_DATA_UPDATE',
                    data: analysis
                }).catch(() => {});
            } catch {}
        }

        // Also notify popup if open
        chrome.runtime.sendMessage({
            type: 'ANALYSIS_UPDATE',
            data: analysis,
        }).catch(() => {});

    } catch (err) {
        console.error('[ANI] Analysis error:', err);
    }
}

// ─── Track Tab Switches ───────────────────────────────────────
chrome.tabs.onActivated.addListener((activeInfo) => {
    if (lastActiveTabId !== null && lastActiveTabId !== activeInfo.tabId) {
        tabSwitches.push(Date.now());
    }
    lastActiveTabId = activeInfo.tabId;
    // Push immediately on tab switch
    pushAnalysis();
});

// Also push on tab create/remove
chrome.tabs.onCreated.addListener(() => pushAnalysis());
chrome.tabs.onRemoved.addListener(() => setTimeout(pushAnalysis, 500));
chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
    if (changeInfo.status === 'complete') pushAnalysis();
});

// ─── Periodic Push ────────────────────────────────────────────
// Use setInterval for 5-second pushes (chrome.alarms min is 1 minute)
setInterval(pushAnalysis, PUSH_INTERVAL_MS);

// Initial push on startup
chrome.runtime.onInstalled.addListener(() => {
    console.log('[ANI] Extension installed — starting automatic tab analysis');
    pushAnalysis();
});

chrome.runtime.onStartup.addListener(() => {
    console.log('[ANI] Extension started — resuming tab analysis');
    pushAnalysis();
});

// ─── Storage ──────────────────────────────────────────────────
async function storeDataPoint(data) {
    try {
        const result = await chrome.storage.local.get('dataPoints');
        const points = result.dataPoints || [];
        // Store compact version (no full tab list)
        points.push({
            timestamp: data.timestamp,
            tabCount: data.tabCount,
            uniqueDomains: data.uniqueDomains,
            categories: data.categories,
            productivityScore: data.productivityScore,
            distractionScore: data.distractionScore,
            switchRate: data.switchRate,
            activeCategory: data.activeTab?.category,
            activeTitle: data.activeTab?.title?.substring(0, 60),
        });

        // Keep last 5000 points (~7 hours at 5-sec intervals)
        if (points.length > 5000) points.splice(0, points.length - 5000);
        await chrome.storage.local.set({ dataPoints: points });
    } catch (e) {
        // Storage might fail silently
    }
}

// ─── Message Handler ──────────────────────────────────────────
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.type === 'GET_TAB_DATA' || msg.type === 'GET_ANALYSIS') {
        analyzeAllTabs().then(data => sendResponse(data));
        return true;
    }

    if (msg.type === 'GET_TAB_COUNT') {
        chrome.tabs.query({}).then(tabs => sendResponse({ tabCount: tabs.length }));
        return true;
    }

    if (msg.type === 'GET_ALL_DATA') {
        chrome.storage.local.get('dataPoints').then(result => {
            sendResponse({ data: result.dataPoints || [] });
        });
        return true;
    }

    if (msg.type === 'CLEAR_DATA') {
        chrome.storage.local.set({ dataPoints: [] }).then(() => sendResponse({ ok: true }));
        return true;
    }
});
