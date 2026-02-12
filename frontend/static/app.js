/**
 * GUARDIAN Transaction Monitoring System
 * Frontend JavaScript Application
 */

// API Configuration
const API_BASE_URL = window.location.origin;
const API_VERSION = '/api/v1';

// DOM Elements
const connectionStatus = document.getElementById('connectionStatus');
const csvFileInput = document.getElementById('csvFileInput');
const pdfFileInput = document.getElementById('pdfFileInput');
const csvUploadArea = document.getElementById('csvUploadArea');
const pdfUploadArea = document.getElementById('pdfUploadArea');
const transactionForm = document.getElementById('transactionForm');
const evaluationResult = document.getElementById('evaluationResult');
const submitBtn = document.getElementById('submitBtn');

// ============================================================
// Connection Status
// ============================================================

async function checkConnection() {
    try {
        const response = await fetch(`${API_BASE_URL}${API_VERSION}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            connectionStatus.classList.add('connected');
            connectionStatus.classList.remove('disconnected');
            connectionStatus.querySelector('.status-text').textContent = 'Connected';
        } else {
            throw new Error('Unhealthy');
        }
    } catch (error) {
        connectionStatus.classList.add('disconnected');
        connectionStatus.classList.remove('connected');
        connectionStatus.querySelector('.status-text').textContent = 'Disconnected';
    }
}

// Check connection on load and every 30 seconds
checkConnection();
setInterval(checkConnection, 30000);

// ============================================================
// Initialize Form with Current DateTime
// ============================================================

function setCurrentDateTime() {
    const now = new Date();
    // Format: YYYY-MM-DDTHH:MM:SS (ISO 8601 format for backend)
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0');
    
    const dateTimeString = `${year}-${month}-${day}T${hours}:${minutes}:${seconds}`;
    
    const dateTimeInput = document.getElementById('transDateTime');
    if (dateTimeInput) {
        dateTimeInput.value = dateTimeString;
    }
}

// Set current datetime when page loads
setCurrentDateTime();

// ============================================================
// System Statistics
// ============================================================

async function refreshStats() {
    try {
        const response = await fetch(`${API_BASE_URL}${API_VERSION}/metrics`);
        const data = await response.json();
        
        document.getElementById('statUsers').textContent = data.total_users || 0;
        document.getElementById('statTransactions').textContent = data.total_transactions || 0;
        document.getElementById('statPolicies').textContent = data.policy_chunks || 0;
    } catch (error) {
        console.error('Failed to refresh stats:', error);
    }
}

// Refresh stats on load
refreshStats();

// ============================================================
// File Upload Handlers
// ============================================================

// Drag and drop for CSV
csvUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    csvUploadArea.classList.add('dragover');
});

csvUploadArea.addEventListener('dragleave', () => {
    csvUploadArea.classList.remove('dragover');
});

csvUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    csvUploadArea.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files).filter(f => f.name.endsWith('.csv'));
    if (files.length > 0) {
        handleCSVUpload(files);
    }
});

csvUploadArea.addEventListener('click', () => {
    csvFileInput.click();
});

csvFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleCSVUpload(Array.from(e.target.files));
    }
});

// Drag and drop for PDF
pdfUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    pdfUploadArea.classList.add('dragover');
});

pdfUploadArea.addEventListener('dragleave', () => {
    pdfUploadArea.classList.remove('dragover');
});

pdfUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    pdfUploadArea.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files).filter(f => f.name.endsWith('.pdf'));
    if (files.length > 0) {
        handlePDFUpload(files);
    }
});

pdfUploadArea.addEventListener('click', () => {
    pdfFileInput.click();
});

pdfFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handlePDFUpload(Array.from(e.target.files));
    }
});

// ============================================================
// CSV Upload Handler
// ============================================================

async function handleCSVUpload(files) {
    const progressDiv = document.getElementById('csvUploadProgress');
    const progressFill = document.getElementById('csvProgressFill');
    const progressText = document.getElementById('csvProgressText');
    const resultDiv = document.getElementById('csvUploadResult');
    
    progressDiv.style.display = 'block';
    resultDiv.innerHTML = '';
    
    let successCount = 0;
    let errorCount = 0;
    
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const progress = ((i + 1) / files.length) * 100;
        
        progressFill.style.width = `${progress}%`;
        progressText.textContent = `Uploading ${file.name}... (${i + 1}/${files.length})`;
        
        // Extract user_id from filename (e.g., "Alice_transactions.csv" -> "Alice")
        const userId = file.name.replace('_transactions.csv', '').replace('.csv', '');
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch(`${API_BASE_URL}${API_VERSION}/users/${userId}/transactions`, {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const data = await response.json();
                successCount++;
                resultDiv.innerHTML += `<div class="success">✅ ${file.name}: ${data.transactions_count || 'Uploaded'} transactions loaded</div>`;
            } else {
                const error = await response.json();
                errorCount++;
                resultDiv.innerHTML += `<div class="error">❌ ${file.name}: ${error.detail || 'Upload failed'}</div>`;
            }
        } catch (error) {
            errorCount++;
            resultDiv.innerHTML += `<div class="error">❌ ${file.name}: ${error.message}</div>`;
        }
    }
    
    progressText.textContent = `Complete: ${successCount} succeeded, ${errorCount} failed`;
    refreshStats();
}

// ============================================================
// PDF Upload Handler
// ============================================================

async function handlePDFUpload(files) {
    const progressDiv = document.getElementById('pdfUploadProgress');
    const progressFill = document.getElementById('pdfProgressFill');
    const progressText = document.getElementById('pdfProgressText');
    const resultDiv = document.getElementById('pdfUploadResult');
    const policyTypeSelect = document.getElementById('policyTypeSelect');
    const policyType = document.getElementById('policyType').value;
    
    progressDiv.style.display = 'block';
    policyTypeSelect.style.display = 'flex';
    resultDiv.innerHTML = '';
    
    let successCount = 0;
    let errorCount = 0;
    
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const progress = ((i + 1) / files.length) * 100;
        
        progressFill.style.width = `${progress}%`;
        progressText.textContent = `Uploading ${file.name}... (${i + 1}/${files.length})`;
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('policy_type', policyType);
        
        try {
            const response = await fetch(`${API_BASE_URL}${API_VERSION}/policies`, {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const data = await response.json();
                successCount++;
                resultDiv.innerHTML += `<div class="success">✅ ${file.name}: ${data.chunks_created || 'Uploaded'} chunks indexed</div>`;
            } else {
                const error = await response.json();
                errorCount++;
                resultDiv.innerHTML += `<div class="error">❌ ${file.name}: ${error.detail || 'Upload failed'}</div>`;
            }
        } catch (error) {
            errorCount++;
            resultDiv.innerHTML += `<div class="error">❌ ${file.name}: ${error.message}</div>`;
        }
    }
    
    progressText.textContent = `Complete: ${successCount} succeeded, ${errorCount} failed`;
    refreshStats();
}

// ============================================================
// Transaction Form Handler
// ============================================================

transactionForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Disable submit button
    submitBtn.disabled = true;
    submitBtn.querySelector('.btn-text').style.display = 'none';
    submitBtn.querySelector('.btn-loader').style.display = 'inline';
    
    // Update hidden datetime field with current time
    setCurrentDateTime();
    
    // Gather form data
    const formData = new FormData(transactionForm);
    
    // Build request payload
    const payload = {
        user_id: formData.get('user_id'),
        cc_num: formData.get('cc_num') || null,
        trans_date_trans_time: formData.get('trans_date_trans_time'),
        merchant: formData.get('merchant'),
        category: 'general',  // Default category since it's not in form
        amt: parseFloat(formData.get('amt')),
        city: formData.get('city'),
        state: formData.get('state'),
        country: 'US'
    };
    
    try {
        const response = await fetch(`${API_BASE_URL}${API_VERSION}/evaluate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        if (response.ok) {
            const result = await response.json();
            displayResult(result);
        } else {
            const error = await response.json();
            // Handle different error formats
            let errorMessage = 'Unknown error';
            if (error.detail) {
                if (Array.isArray(error.detail)) {
                    // Validation errors from FastAPI
                    errorMessage = error.detail.map(e => {
                        const field = Array.isArray(e.loc) ? e.loc.join('.') : 'field';
                        const msg = e.msg || 'validation error';
                        return `${field}: ${msg}`;
                    }).join('\n');
                } else if (typeof error.detail === 'object') {
                    errorMessage = JSON.stringify(error.detail, null, 2);
                } else {
                    errorMessage = error.detail;
                }
            }
            console.error('Full error:', error);
            alert(`Evaluation failed:\n${errorMessage}`);
        }
    } catch (error) {
        console.error('Request error:', error);
        alert(`Error: ${error.message}`);
    } finally {
        // Re-enable submit button
        submitBtn.disabled = false;
        submitBtn.querySelector('.btn-text').style.display = 'inline';
        submitBtn.querySelector('.btn-loader').style.display = 'none';
    }
});

// ============================================================
// Display Evaluation Result
// ============================================================

function displayResult(result) {
    evaluationResult.style.display = 'block';
    
    // Decision badge
    const decisionBadge = document.getElementById('decisionBadge');
    decisionBadge.textContent = result.decision;
    decisionBadge.className = `decision-badge ${result.decision}`;
    
    // Scores
    const riskScore = Math.round(result.fused_score * 100);
    const behavioralScore = Math.round(result.behavioral_score * 100);
    const policyScore = Math.round(result.policy_score * 100);
    
    document.getElementById('riskScoreFill').style.width = `${riskScore}%`;
    document.getElementById('riskScoreValue').textContent = `${riskScore}%`;
    
    document.getElementById('behavioralScoreFill').style.width = `${behavioralScore}%`;
    document.getElementById('behavioralScoreValue').textContent = `${behavioralScore}%`;
    
    document.getElementById('policyScoreFill').style.width = `${policyScore}%`;
    document.getElementById('policyScoreValue').textContent = `${policyScore}%`;
    
    // Explanation
    document.getElementById('explanationText').textContent = result.explanation;
    
    // Behavioral evidence
    const behavioralEvidence = document.getElementById('behavioralEvidence');
    behavioralEvidence.innerHTML = '';
    
    if (result.evidence && result.evidence.behavioral_rag) {
        const behEvidence = result.evidence.behavioral_rag;
        
        if (behEvidence.deviations && behEvidence.deviations.length > 0) {
            behEvidence.deviations.forEach(deviation => {
                behavioralEvidence.innerHTML += `<li>⚠️ ${deviation}</li>`;
            });
        }
        
        if (behEvidence.similar_transactions && behEvidence.similar_transactions.length > 0) {
            behavioralEvidence.innerHTML += `<li>📊 Found ${behEvidence.similar_transactions.length} similar transactions</li>`;
        }
        
        if (behavioralEvidence.innerHTML === '') {
            behavioralEvidence.innerHTML = '<li>✅ No behavioral anomalies detected</li>';
        }
    }
    
    // Policy evidence
    const policyEvidence = document.getElementById('policyEvidence');
    policyEvidence.innerHTML = '';
    
    if (result.evidence && result.evidence.policy_rag) {
        const polEvidence = result.evidence.policy_rag;
        
        if (polEvidence.violations && polEvidence.violations.length > 0) {
            polEvidence.violations.forEach(violation => {
                policyEvidence.innerHTML += `<li>🚫 ${violation}</li>`;
            });
        }
        
        if (polEvidence.retrieved_policies && polEvidence.retrieved_policies.length > 0) {
            policyEvidence.innerHTML += `<li>📋 Checked ${polEvidence.retrieved_policies.length} policy documents</li>`;
        }
        
        if (policyEvidence.innerHTML === '') {
            policyEvidence.innerHTML = '<li>✅ No policy violations detected</li>';
        }
    }
    
    // Transaction meta
    document.getElementById('transactionId').textContent = result.transaction_id || 'N/A';
    document.getElementById('processingTime').textContent = 
        result.processing_time_ms ? `${result.processing_time_ms.toFixed(2)}ms` : 'N/A';
    
    // Scroll to result
    evaluationResult.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ============================================================
// Format Credit Card Input
// ============================================================

document.getElementById('ccNumber').addEventListener('input', (e) => {
    let value = e.target.value.replace(/\D/g, '');
    if (value.length > 16) value = value.slice(0, 16);
    
    // Add dashes every 4 digits
    const formatted = value.match(/.{1,4}/g)?.join('-') || value;
    e.target.value = formatted;
});

console.log('GUARDIAN Frontend initialized');
