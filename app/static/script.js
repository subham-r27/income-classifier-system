// Load unique values for dropdowns on page load
document.addEventListener('DOMContentLoaded', async () => {
    await loadUniqueValues();
});

// Load unique values from API
async function loadUniqueValues() {
    try {
        const response = await fetch('/api/unique-values');
        const data = await response.json();
        
        // Populate dropdowns
        populateDropdown('JobType', data.JobType || []);
        populateDropdown('EdType', data.EdType || []);
        populateDropdown('maritalstatus', data.maritalstatus || []);
        populateDropdown('occupation', data.occupation || []);
        populateDropdown('relationship', data.relationship || []);
        populateDropdown('race', data.race || []);
        populateDropdown('gender', data.gender || []);
        populateDropdown('nativecountry', data.nativecountry || []);
        
    } catch (error) {
        console.error('Error loading unique values:', error);
        showError('Failed to load form options. Please refresh the page.');
    }
}

// Populate a dropdown with options
function populateDropdown(selectId, options) {
    const select = document.getElementById(selectId);
    if (!select) return;
    
    // Clear existing options except the first one
    const firstOption = select.options[0];
    select.innerHTML = '';
    select.appendChild(firstOption);
    
    // Add options
    options.forEach(option => {
        const optionElement = document.createElement('option');
        optionElement.value = option;
        optionElement.textContent = option;
        select.appendChild(optionElement);
    });
}

// Handle form submission
document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Hide previous results/errors
    document.getElementById('result').style.display = 'none';
    document.getElementById('error').style.display = 'none';
    
    // Get form data
    const formData = {
        age: parseInt(document.getElementById('age').value),
        JobType: document.getElementById('JobType').value,
        EdType: document.getElementById('EdType').value,
        maritalstatus: document.getElementById('maritalstatus').value,
        occupation: document.getElementById('occupation').value,
        relationship: document.getElementById('relationship').value,
        race: document.getElementById('race').value,
        gender: document.getElementById('gender').value,
        capitalgain: parseInt(document.getElementById('capitalgain').value) || 0,
        capitalloss: parseInt(document.getElementById('capitalloss').value) || 0,
        hoursperweek: parseInt(document.getElementById('hoursperweek').value),
        nativecountry: document.getElementById('nativecountry').value,
        model_type: document.getElementById('modelType').value
    };
    
    // Validate model selection
    if (!formData.model_type) {
        showError('Please select a model (KNN or Logistic Regression)');
        return;
    }
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('submitBtn').disabled = true;
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Prediction failed');
        }
        
        const result = await response.json();
        showResult(result);
        
    } catch (error) {
        console.error('Prediction error:', error);
        showError(error.message || 'An error occurred while making the prediction');
    } finally {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('submitBtn').disabled = false;
    }
});

// Show prediction result
function showResult(result) {
    const resultContainer = document.getElementById('result');
    const resultContent = document.getElementById('resultContent');
    
    // Remove previous classes
    resultContainer.classList.remove('success', 'low');
    
    // Add appropriate class based on prediction
    if (result.prediction === 1) {
        resultContainer.classList.add('success');
    } else {
        resultContainer.classList.add('low');
    }
    
    const modelName = result.model_used === 'knn' ? 'K-Nearest Neighbors' : 'Logistic Regression';
    const incomeText = result.prediction === 1 
        ? 'greater than $50,000' 
        : 'less than or equal to $50,000';
    
    resultContent.innerHTML = `
        <p><strong>Predicted Income:</strong> ${incomeText}</p>
        <p style="margin-top: 10px; font-size: 0.9em; opacity: 0.9;">
            Model Used: ${modelName}
        </p>
    `;
    
    resultContainer.style.display = 'block';
    
    // Scroll to result
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Show error message
function showError(message) {
    const errorContainer = document.getElementById('error');
    const errorMessage = document.getElementById('errorMessage');
    
    errorMessage.textContent = message;
    errorContainer.style.display = 'block';
    
    // Scroll to error
    errorContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Handle form reset
document.getElementById('resetBtn').addEventListener('click', () => {
    document.getElementById('result').style.display = 'none';
    document.getElementById('error').style.display = 'none';
});
