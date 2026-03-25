/**
 * ANI Flow Data Collector — Content Script
 * Runs on every page to collect screen-level context.
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

    if (msg.type === 'CAPTURE_VISIBLE') {
        // Request a screenshot from the background script
        chrome.runtime.sendMessage({ type: 'CAPTURE_SCREEN' }, (response) => {
            sendResponse(response);
        });
        return true;
    }
});

// ─── Periodic context reporting ───────────────────────────────
// Reports context to background every 5 minutes for enriched data collection
let lastReport = 0;
const REPORT_INTERVAL = 5 * 60 * 1000; // 5 minutes

function maybeReport() {
    const now = Date.now();
    if (now - lastReport < REPORT_INTERVAL) return;
    lastReport = now;

    const context = extractPageContext();
    chrome.runtime.sendMessage({
        type: 'PAGE_CONTEXT_UPDATE',
        data: context,
    }).catch(() => {}); // Background may not be listening
}

// Check periodically
setInterval(maybeReport, 60000);
// Also on visibility change
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') maybeReport();
});
