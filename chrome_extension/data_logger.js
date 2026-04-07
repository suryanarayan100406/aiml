/**
 * ANI Tab Analyzer — Data Logger (Legacy compat)
 * Kept for CSV export functionality. Self-report removed.
 */
const DataLogger = {
    /** Export stored analysis data as CSV */
    async downloadCSV() {
        const response = await new Promise(resolve => {
            chrome.runtime.sendMessage({ type: 'GET_ALL_DATA' }, resolve);
        });
        const data = response?.data || [];
        if (data.length === 0) { alert('No data to export'); return; }

        const allKeys = new Set();
        data.forEach(d => Object.keys(d).forEach(k => {
            if (typeof d[k] !== 'object') allKeys.add(k);
        }));
        const headers = [...allKeys];
        const rows = data.map(d =>
            headers.map(h => {
                const val = d[h] ?? '';
                return typeof val === 'string' && val.includes(',') ? `"${val}"` : val;
            }).join(',')
        );
        const csv = [headers.join(','), ...rows].join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `ani_tab_analysis_${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
        URL.revokeObjectURL(url);
    },
};

if (typeof window !== 'undefined') window.DataLogger = DataLogger;
