window.addEventListener('load', function () {
    let triageObject = {
        name: document.getElementById("name").textContent,
        message: document.getElementById("message").textContent,
        status: document.getElementById("status").textContent
    }
    
    let id = Math.floor(Math.random() * (999999 - 100000) + 100000);
    let case_id = `case${id}`;

    let historyContainer = document.getElementById("historyContainer");    
    let history = JSON.parse(localStorage.getItem('triageHistory'));
    
    if (history == null) {
        history = {};
    }
    history[case_id] = triageObject;

    localStorage.setItem("triageHistory", JSON.stringify(history));
    
    // Update history display
    Object.entries(history).forEach(([caseId, data]) => {
        if (caseId !== case_id) {  // Don't duplicate current case
            const historyItem = document.createElement('div');
            historyItem.className = 'historyItem';
            historyItem.innerHTML = `
                <div class="historyContent">
                    <h1>${data.name} - <span style="font-weight: bold;">${data.status}</span></h1>
                    <p><span style="font-style: italic;">"${data.message}"</span></p>
                </div>
            `;
            historyContainer.appendChild(historyItem);
        }
    });
});