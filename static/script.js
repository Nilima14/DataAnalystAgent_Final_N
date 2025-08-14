// Data Analyst Agent Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('analysisForm');
    const submitBtn = document.getElementById('submitBtn');
    const submitText = document.getElementById('submitText');
    const submitSpinner = document.getElementById('submitSpinner');
    const resultsSection = document.getElementById('resultsSection');
    const resultsContent = document.getElementById('resultsContent');
    const errorSection = document.getElementById('errorSection');
    const errorContent = document.getElementById('errorContent');

    // Form submission handler
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Reset previous results
        hideResults();
        hideError();
        
        // Show loading state
        showLoading();
        
        try {
            // Prepare form data
            const formData = new FormData();
            
            // Add questions file
            const questionFile = document.getElementById('questionFile').files[0];
            if (!questionFile) {
                throw new Error('Questions file is required');
            }
            
            // Ensure the file is named questions.txt
            const questionsBlob = new Blob([await questionFile.arrayBuffer()], { type: 'text/plain' });
            formData.append('files', questionsBlob, 'questions.txt');
            
            // Add additional files
            const additionalFiles = document.getElementById('additionalFiles').files;
            for (let i = 0; i < additionalFiles.length; i++) {
                formData.append('files', additionalFiles[i], additionalFiles[i].name);
            }
            
            // Make API request
            const response = await fetch('/api/', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            displayResults(result);
            
        } catch (error) {
            console.error('Analysis failed:', error);
            showError(error.message);
        } finally {
            hideLoading();
        }
    });
    
    function showLoading() {
        submitBtn.disabled = true;
        submitText.textContent = '🔄 Analyzing...';
        submitSpinner.classList.remove('d-none');
        
        // Create and show loading overlay
        const overlay = document.createElement('div');
        overlay.id = 'loadingOverlay';
        overlay.className = 'loading-overlay';
        overlay.innerHTML = `
            <div class="loading-content">
                <div class="loading-spinner mb-3"></div>
                <h5>🤖 AI is analyzing your data...</h5>
                <p class="text-muted">This may take up to 3 minutes</p>
                <div id="loadingStatus" class="mt-3">
                    <small class="text-info">⏳ Processing request...</small>
                </div>
            </div>
        `;
        document.body.appendChild(overlay);
        
        // Update loading status messages
        let statusIndex = 0;
        const statusMessages = [
            '⏳ Processing request...',
            '🔍 Breaking down the task...',
            '💻 Generating Python code...',
            '🚀 Executing analysis...',
            '📊 Processing results...'
        ];
        
        const statusInterval = setInterval(() => {
            const statusElement = document.getElementById('loadingStatus');
            if (statusElement && statusIndex < statusMessages.length - 1) {
                statusIndex++;
                statusElement.innerHTML = `<small class="text-info">${statusMessages[statusIndex]}</small>`;
            }
        }, 15000); // Update every 15 seconds
        
        overlay.statusInterval = statusInterval;
    }
    
    function hideLoading() {
        submitBtn.disabled = false;
        submitText.textContent = '🚀 Start Analysis';
        submitSpinner.classList.add('d-none');
        
        // Remove loading overlay
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            if (overlay.statusInterval) {
                clearInterval(overlay.statusInterval);
            }
            overlay.remove();
        }
    }
    
    function displayResults(data) {
        try {
            let resultsHtml = '';
            
            if (Array.isArray(data)) {
                // Handle JSON array response
                resultsHtml = '<h6 class="mb-3">📋 Analysis Results (JSON Array)</h6>';
                resultsHtml += `<div class="results-json">${JSON.stringify(data, null, 2)}</div>`;
                
                // Check for images in the array
                data.forEach((item, index) => {
                    if (typeof item === 'string' && item.startsWith('data:image/')) {
                        resultsHtml += `
                            <div class="mt-3">
                                <h6>🖼️ Visualization ${index + 1}</h6>
                                <img src="${item}" alt="Generated visualization" class="results-image">
                            </div>
                        `;
                    }
                });
                
            } else if (typeof data === 'object' && data !== null) {
                // Handle JSON object response
                resultsHtml = '<h6 class="mb-3">📊 Analysis Results (JSON Object)</h6>';
                resultsHtml += `<div class="results-json">${JSON.stringify(data, null, 2)}</div>`;
                
                // Check for images in object values
                Object.entries(data).forEach(([key, value]) => {
                    if (typeof value === 'string' && value.startsWith('data:image/')) {
                        resultsHtml += `
                            <div class="mt-3">
                                <h6>🖼️ ${key}</h6>
                                <img src="${value}" alt="${key}" class="results-image">
                            </div>
                        `;
                    }
                });
                
            } else {
                // Handle simple string/number response
                resultsHtml = '<h6 class="mb-3">📝 Analysis Result</h6>';
                resultsHtml += `<div class="results-json">${JSON.stringify(data, null, 2)}</div>`;
            }
            
            resultsContent.innerHTML = resultsHtml;
            resultsSection.classList.remove('d-none');
            resultsSection.classList.add('fade-in');
            
            // Scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth' });
            
        } catch (error) {
            console.error('Error displaying results:', error);
            showError('Failed to display results: ' + error.message);
        }
    }
    
    function showError(message) {
        errorContent.innerHTML = `<strong>Details:</strong><br>${escapeHtml(message)}`;
        errorSection.classList.remove('d-none');
        errorSection.classList.add('fade-in');
        errorSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    function hideResults() {
        resultsSection.classList.add('d-none');
        resultsSection.classList.remove('fade-in');
    }
    
    function hideError() {
        errorSection.classList.add('d-none');
        errorSection.classList.remove('fade-in');
    }
    
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    // File input validation
    document.getElementById('questionFile').addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file && !file.name.endsWith('.txt')) {
            alert('Please select a .txt file for questions');
            e.target.value = '';
        }
    });
    
    // Drag and drop functionality
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        const parent = input.closest('.mb-3');
        
        parent.addEventListener('dragover', function(e) {
            e.preventDefault();
            parent.classList.add('border-primary');
        });
        
        parent.addEventListener('dragleave', function(e) {
            e.preventDefault();
            parent.classList.remove('border-primary');
        });
        
        parent.addEventListener('drop', function(e) {
            e.preventDefault();
            parent.classList.remove('border-primary');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                input.files = files;
                
                // Trigger change event
                const event = new Event('change', { bubbles: true });
                input.dispatchEvent(event);
            }
        });
    });
    
    // Health check on page load
    checkHealth();
    
    async function checkHealth() {
        try {
            const response = await fetch('/health');
            if (response.ok) {
                console.log('✅ API health check passed');
            } else {
                console.warn('⚠️ API health check failed');
            }
        } catch (error) {
            console.error('❌ API health check error:', error);
        }
    }
});
