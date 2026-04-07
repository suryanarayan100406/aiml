/**
 * ANI Tab Analyzer — Content Script
 * Bridges tab analysis data from the Chrome extension background
 * to the ANI frontend dashboard via postMessage.
 *
 * Runs on every page. Actively pushes data when ANI dashboard is detected.
 */

// ─── Listen for background pushes ─────────────────────────────
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.type === 'TAB_DATA_UPDATE') {
        // Forward to page via postMessage so ANI frontend JS can read it securely
        // postMessage safely crosses the isolated world boundary
        window.postMessage({ type: 'ANI_TAB_DATA', data: msg.data }, '*');
        sendResponse({ ok: true });
        return true;
    }

    if (msg.type === 'GET_PAGE_CONTEXT') {
        sendResponse(extractPageContext());
        return true;
    }
});

// ─── Listen for requests FROM the ANI frontend page ───────────
window.addEventListener('message', (e) => {
    // Only accept messages from ourselves
    if (e.source !== window || !e.data || e.data.type !== 'ANI_REQUEST_TABS') return;
    
    chrome.runtime.sendMessage({ type: 'GET_TAB_DATA' }, (response) => {
        if (chrome.runtime.lastError) {
            console.warn('[ANI Extension] Background script unresponsive:', chrome.runtime.lastError);
            return;
        }
        if (response) {
            window.postMessage({ type: 'ANI_TAB_DATA', data: response }, '*');
        }
    });
});

// ─── Detect ANI Dashboard and auto-push ───────────────────────
const isAniPage = document.title.includes('ANI') ||
                   window.location.href.includes('ani-flow-optimizer') ||
                   window.location.href.includes('localhost:8080') ||
                   window.location.href.includes('127.0.0.1:8080');

if (isAniPage) {
    console.log('[ANI Extension] Dashboard detected — starting auto tab push');

    // Push immediately
    requestTabData();

    // Then every 3 seconds (making it slightly faster for responsiveness)
    setInterval(requestTabData, 3000);
}

function requestTabData() {
    chrome.runtime.sendMessage({ type: 'GET_TAB_DATA' }, (response) => {
        if (chrome.runtime.lastError) return;
        if (response) {
            window.postMessage({ type: 'ANI_TAB_DATA', data: response }, '*');
        }
    });
}

// ─── Page Context Extraction ──────────────────────────────────
function extractPageContext() {
    return {
        title: document.title,
        url: window.location.href,
        domain: window.location.hostname,
        hasVideo: document.querySelectorAll('video').length > 0,
        hasAudio: document.querySelectorAll('audio').length > 0,
        isInputFocused: ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName),
        timestamp: Date.now(),
    };
}
