/**
 * ANI Flow Data Collector — Content Script
 * Bridges real-time tab data from the Chrome extension to the ANI frontend.
 * Runs on every page, but actively pushes data only when ANI dashboard is open.
 */

// ─── Page Context Extraction ──────────────────────────────────
function extractPageContext() {
    return {
        title: document.title,
        url: window.location.href,
        domain: window.location.hostname,
        hasVideo: document.querySelectorAll('video').length > 0,
        hasAudio: document.querySelectorAll('audio').length > 0,
        textContentLength: document.body?.innerText?.length || 0,
        scrollPosition: window.scrollY,
        pageHeight: document.documentElement.scrollHeight,
        focusedElement: document.activeElement?.tagName || 'NONE',
        isInputFocused: ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName),
        timestamp: Date.now(),
    };
}

// ─── Listen for requests from background/popup ────────────────
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.type === 'GET_PAGE_CONTEXT') {
        sendResponse(extractPageContext());
        return true;
    }

    // Background sends us tab data to push to the page
    if (msg.type === 'TAB_DATA_UPDATE') {
        // Push to the page via CustomEvent so ANI frontend can read it
        window.dispatchEvent(new CustomEvent('ani-tab-data', {
            detail: msg.data
        }));
        sendResponse({ ok: true });
        return true;
    }
});

// ─── Listen for requests FROM the ANI frontend page ───────────
// The frontend page dispatches 'ani-request-tabs' when it needs tab data
window.addEventListener('ani-request-tabs', () => {
    chrome.runtime.sendMessage({ type: 'GET_TAB_DATA' }, (response) => {
        if (chrome.runtime.lastError) {
            console.warn('[ANI Extension] No response from background:', chrome.runtime.lastError.message);
            return;
        }
        if (response) {
            window.dispatchEvent(new CustomEvent('ani-tab-data', {
                detail: response
            }));
        }
    });
});

// ─── Auto-push tab data every 10 seconds when ANI is on this page ──
// Detect if this is the ANI dashboard page
const isAniPage = document.title.includes('ANI') || 
                   document.querySelector('#ani-avatar') !== null ||
                   window.location.href.includes('ani-flow-optimizer');

if (isAniPage) {
    console.log('[ANI Extension] Dashboard detected, starting tab data push');
    
    // Push immediately
    requestTabData();
    
    // Then every 10 seconds
    setInterval(requestTabData, 10000);
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

// ─── Periodic context reporting ───────────────────────────────
let lastReport = 0;
const REPORT_INTERVAL = 5 * 60 * 1000;

function maybeReport() {
    const now = Date.now();
    if (now - lastReport < REPORT_INTERVAL) return;
    lastReport = now;

    const context = extractPageContext();
    chrome.runtime.sendMessage({
        type: 'PAGE_CONTEXT_UPDATE',
        data: context,
    }).catch(() => {});
}

setInterval(maybeReport, 60000);
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') maybeReport();
});
