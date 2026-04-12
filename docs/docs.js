const DOCUMENTS = [
    { id: 'README.md', title: 'README' },
    { id: 'USAGE_GUIDE.md', title: 'USAGE GUIDE' }
];

const navContainer = document.getElementById('doc-nav');
const contentContainer = document.getElementById('markdown-content');

// Helper to fetch and render a markdown file
async function loadDocument(filename) {
    try {
        contentContainer.innerHTML = '<p class="loading-text">LOADING...</p>';
        const response = await fetch(`/${filename}`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const text = await response.text();
        
        // Parse markdown to HTML
        contentContainer.innerHTML = marked.parse(text);
    } catch (error) {
        console.error('Error loading markdown:', error);
        contentContainer.innerHTML = `
            <h2>ERROR LOADING DOCUMENT</h2>
            <p>Could not fetch /${filename}. Ensure the local server is running.</p>
        `;
    }
}

// Initialize navigation
function initNav() {
    DOCUMENTS.forEach((doc, index) => {
        const navItem = document.createElement('div');
        navItem.className = 'nav-item';
        navItem.textContent = doc.title;
        
        navItem.addEventListener('click', () => {
            // Update active state
            document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
            navItem.classList.add('active');
            
            // Load content
            loadDocument(doc.id);
            
            // Update URL hash without scrolling
            history.replaceState(null, null, `#${doc.id}`);
        });
        
        navContainer.appendChild(navItem);
    });

    // Handle initial load based on URL hash
    const initialDocId = window.location.hash.substring(1);
    const initialDocIndex = DOCUMENTS.findIndex(d => d.id === initialDocId);
    
    if (initialDocIndex !== -1) {
        navContainer.children[initialDocIndex].click();
    } else if (DOCUMENTS.length > 0) {
        navContainer.children[0].click();
    }
}

// Start
document.addEventListener('DOMContentLoaded', () => {
    // Configure marked options if needed
    marked.setOptions({
        gfm: true,
        breaks: true
    });
    
    initNav();
});
