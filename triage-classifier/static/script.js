// Function to handle triage history
window.addEventListener('load', function () {
    const triageData = {
        name: document.getElementById("name")?.textContent,
        message: document.getElementById("message")?.textContent,
        status: document.getElementById("status")?.textContent,
        timestamp: new Date().toISOString()
    };
    
    // Only proceed if we have valid triage data
    if (triageData.name && triageData.message && triageData.status) {
        // Generate unique case ID using timestamp and random number
        const id = `case_${Date.now()}_${Math.floor(Math.random() * 1000)}`;
        
        // Get existing history or initialize new object
        let history = JSON.parse(localStorage.getItem('triageHistory')) || {};
        
        // Add new case to history (limit to most recent 50 cases)
        history[id] = triageData;
        const sortedKeys = Object.keys(history).sort().reverse();
        if (sortedKeys.length > 50) {
            const keysToRemove = sortedKeys.slice(50);
            keysToRemove.forEach(key => delete history[key]);
        }
        
        // Save updated history
        localStorage.setItem('triageHistory', JSON.stringify(history));
        
        // Update history display
        updateHistoryDisplay(history, id);
    }
});

// Function to update history display
function updateHistoryDisplay(history, currentId) {
    const historyContainer = document.getElementById("historyContainer");
    if (!historyContainer) return;
    
    // Clear existing history items (except the current case)
    const currentCase = historyContainer.firstElementChild;
    historyContainer.innerHTML = '';
    if (currentCase) {
        historyContainer.appendChild(currentCase);
    }
    
    // Add history header
    const headerElement = document.createElement('h1');
    headerElement.className = 'historyHeader';
    headerElement.textContent = 'Status History';
    historyContainer.appendChild(headerElement);
    
    // Sort history by timestamp (most recent first)
    const sortedHistory = Object.entries(history)
        .sort(([, a], [, b]) => new Date(b.timestamp) - new Date(a.timestamp))
        .slice(0, 10); // Show only last 10 entries
    
    // Add history items
    sortedHistory.forEach(([caseId, data]) => {
        if (caseId !== currentId) {
            const historyItem = document.createElement('div');
            historyItem.className = 'historyItem';
            historyItem.innerHTML = `
                <div class="historyContent">
                    <h1>${data.name} - <span style="font-weight: bold;">${data.status}</span></h1>
                    <p><span style="font-style: italic;">"${data.message}"</span></p>
                    <small class="timestamp">${new Date(data.timestamp).toLocaleString()}</small>
                </div>
            `;
            historyContainer.appendChild(historyItem);
        }
    });
}

// Function to clear history
function clearTriageHistory() {
    if (confirm('Are you sure you want to clear all triage history?')) {
        localStorage.removeItem('triageHistory');
        const historyContainer = document.getElementById("historyContainer");
        if (historyContainer) {
            // Preserve only the current case
            const currentCase = historyContainer.querySelector('.historyItem');
            historyContainer.innerHTML = '';
            if (currentCase) {
                historyContainer.appendChild(currentCase);
            }
        }
    }
}