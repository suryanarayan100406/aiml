/**
 * ANI Flow Data Collector — Data Logger
 * Manages data collection, self-report prompts, and CSV export.
 */

const DataLogger = {
    /** Get all collected data points */
    async getAll() {
        return new Promise((resolve) => {
            chrome.runtime.sendMessage({ type: 'GET_ALL_DATA' }, (response) => {
                resolve(response?.data || []);
            });
        });
    },

    /** Save a self-report entry */
    async saveSelfReport(focusScore, taskDescription = '') {
        const tabCountResponse = await new Promise((resolve) => {
            chrome.runtime.sendMessage({ type: 'GET_TAB_COUNT' }, resolve);
        });

        const report = {
            type: 'self_report',
            focusScore: focusScore,        // 1-5 scale
            taskDescription: taskDescription,
            tabCount: tabCountResponse?.tabCount || 0,
            flowStateLabel: focusScore <= 2 ? 'DISTRACTED' :
                           focusScore === 3 ? 'SOFT_FLOW' :
                           focusScore === 4 ? 'DEEP_FLOW' : 'PEAK_FLOW',
        };

        return new Promise((resolve) => {
            chrome.runtime.sendMessage({ type: 'SAVE_SELF_REPORT', data: report }, resolve);
        });
    },

    /** Export all data as CSV */
    async exportCSV() {
        const data = await this.getAll();
        if (data.length === 0) return '';

        // Get all unique keys
        const allKeys = new Set();
        data.forEach(d => Object.keys(d).forEach(k => allKeys.add(k)));
        const headers = [...allKeys];

        const rows = data.map(d =>
            headers.map(h => {
                const val = d[h] ?? '';
                return typeof val === 'string' && val.includes(',') ? `"${val}"` : val;
            }).join(',')
        );

        return [headers.join(','), ...rows].join('\n');
    },

    /** Download CSV file */
    async downloadCSV() {
        const csv = await this.exportCSV();
        if (!csv) return;

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `ani_flow_data_${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
        URL.revokeObjectURL(url);
    },

    /** Clear all stored data */
    async clearAll() {
        return new Promise((resolve) => {
            chrome.runtime.sendMessage({ type: 'CLEAR_DATA' }, resolve);
        });
    },

    /** Get summary statistics */
    async getSummary() {
        const data = await this.getAll();
        if (data.length === 0) return { total: 0 };

        const selfReports = data.filter(d => d.type === 'self_report');
        const autoCollected = data.filter(d => d.type !== 'self_report');

        const tabCounts = data.filter(d => d.tabCount !== undefined).map(d => d.tabCount);
        const avgTabs = tabCounts.length > 0 ? tabCounts.reduce((a, b) => a + b) / tabCounts.length : 0;

        const focusScores = selfReports.map(d => d.focusScore).filter(Boolean);
        const avgFocus = focusScores.length > 0 ? focusScores.reduce((a, b) => a + b) / focusScores.length : 0;

        return {
            total: data.length,
            selfReports: selfReports.length,
            autoCollected: autoCollected.length,
            avgTabs: Math.round(avgTabs * 10) / 10,
            avgFocus: Math.round(avgFocus * 10) / 10,
            oldestEntry: data[0]?.timestamp || 'N/A',
            newestEntry: data[data.length - 1]?.timestamp || 'N/A',
        };
    },
};

// Export for popup
if (typeof window !== 'undefined') {
    window.DataLogger = DataLogger;
}
