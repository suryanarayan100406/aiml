/**
 * ANI Tab Analyzer — Content Script
 * Bridges tab analysis data from the Chrome extension background
 * to the ANI frontend dashboard via CustomEvents.
 *
 * Runs on every page. Actively pushes data when ANI dashboard is detected.
 */

// ─── Listen for background pushes ─────────────────────────────
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.type === 'TAB_DATA_UPDATE') {
        // Forward to page via CustomEvent so ANI frontend JS can read it
        window.dispatchEvent(new CustomEvent('ani-tab-data', {
            detail: msg.data
        }));
        sendResponse({ ok: true });
        return true;
    }

    if (msg.type === 'GET_PAGE_CONTEXT') {
        sendResponse(extractPageContext());
        return true;
    }
});

// ─── Listen for requests FROM the ANI frontend page ───────────
window.addEventListener('ani-request-tabs', () => {
    chrome.runtime.sendMessage({ type: 'GET_TAB_DATA' }, (response) => {
        if (chrome.runtime.lastError) return;
        if (response) {
            window.dispatchEvent(new CustomEvent('ani-tab-data', {
                detail: response
            }));
        }
    });
});

// ─── Detect ANI Dashboard and auto-push ───────────────────────
const isAniPage = document.title.includes('ANI') ||
                   window.location.href.includes('ani-flow-optimizer') ||
                   window.location.href.includes('localhost:8080');

if (isAniPage) {
    console.log('[ANI Extension] Dashboard detected — starting auto tab push');

    // Push immediately
    requestTabData();

    // Then every 5 seconds (matching background interval)
    setInterval(requestTabData, 5000);
}

function requestTabData() {
    chrome.runtime.sendMessage({ type: 'GET_TAB_DATA' }, (response) => {
        if (chrome.runtime.lastError) return;
        if (response) {
            window.dispatchEvent(new CustomEvent('ani-tab-data', {
                detail: response
            }));
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
