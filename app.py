import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
from flask import Flask, request, render_template_string, flash, make_response, send_from_directory
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil

# Flask app configuration
app = Flask(__name__)
app.secret_key = 'secure-secret-key'
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Ensure static/images directory exists
os.makedirs(os.path.join(app.root_path, 'static', 'images'), exist_ok=True)

# Categorical feature options (for validation)
CATEGORICAL_OPTIONS = {
    'Sex': ['0', '1'],
    'CP': ['0', '1', '2', '3'],
    'Fbs': ['0', '1'],
    'Restecg': ['0', '1', '2'],
    'Exang': ['0', '1'],
    'Slope': ['0', '1', '2'],
    'CA': ['0', '1', '2', '3', '4'],
    'Thal': ['0', '1', '2', '3']
}

# Base HTML template
BASE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Open+Sans:wght@400;600&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --primary: #3498db;
            --secondary: #e74c3c;
            --success: #2ecc71;
            --warning: #f39c12;
            --info: #9b59b6;
            --background: #121212;
            --text: #F5F5F5;
            --accent: #BB86FC;
            --sidebar-width: 220px;
            --sidebar-bg: rgba(30, 30, 50, 0.8);
            --container-bg: rgba(40, 40, 60, 0.7); 
        }
        
        body {
            font-family: 'Roboto', sans-serif;
            background: var(--background);
            color: var(--text);
            min-height: 100vh;
            position: relative;
            line-height: 1.6;
        }
        
        #particles-js {
            position: fixed;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, #1f1b4e, #3f51b5);
            z-index: -1;
        }
        
        .sidebar {
            width: var(--sidebar-width);
            height: 100vh;
            position: fixed;
            top: 0;
            left: 0;
            background: var(--sidebar-bg);
            backdrop-filter: blur(15px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
            padding: 25px 20px;
            box-shadow: 2px 0 10px rgba(0, 0, 0, 0.3);
            z-index: 10;
            transition: transform 0.3s ease;
        }
        
        .sidebar-header {
            display: flex;
            align-items: center;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .sidebar-header svg {
            margin-right: 10px;
            fill: var(--accent);
        }
        
        .sidebar h3 {
            color: var(--accent);
            font-size: 1.4rem;
            font-weight: 500;
            letter-spacing: 0.5px;
        }
        
        .nav-links {
            margin-top: 20px;
        }
        
        .sidebar a {
            display: flex;
            align-items: center;
            color: var(--text);
            text-decoration: none;
            padding: 12px 15px;
            margin-bottom: 12px;
            border-radius: 8px;
            transition: all 0.3s;
            font-weight: 400;
        }
        
        .sidebar a svg {
            margin-right: 10px;
            width: 20px;
            height: 20px;
        }
        
        .sidebar a:hover {
            background: rgba(255, 255, 255, 0.15);
            transform: translateX(5px);
        }
        
        .sidebar a.active {
            background: rgba(187, 134, 252, 0.2);
            border-left: 3px solid var(--accent);
        }
        
        .mobile-toggle {
            display: none;
            position: fixed;
            top: 15px;
            left: 15px;
            z-index: 100;
            background: var(--container-bg);
            border-radius: 50%;
            width: 45px;
            height: 45px;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            border: none;
            color: var(--text);
        }
        
        .main-content {
            margin-left: var(--sidebar-width);
            padding: 30px;
            max-width: 1200px;
            transition: margin 0.3s ease;
        }
        
        .page-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
        }
        
        .page-header h1 {
            font-size: 2rem;
            font-weight: 500;
            color: var(--text);
            position: relative;
        }
        
        .page-header h1::after {
            content: '';
            position: absolute;
            bottom: -5px;
            left: 0;
            width: 60px;
            height: 3px;
            background: var(--accent);
            border-radius: 3px;
        }
        
        .glass-container {
            background: var(--container-bg);
            backdrop-filter: blur(15px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            margin-bottom: 25px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .glass-container:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
        }
        
        .flash-message {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            text-align: center;
            color: var(--text);
            animation: fadeIn 0.5s;
        }
        
        .flash-message.error {
            border-color: var(--secondary);
            color: var(--secondary);
        }
        
        .flash-message.success {
            border-color: var(--success);
            color: var(--success);
        }
        
        input, select, textarea, button {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 6px;
            padding: 10px 15px;
            color: var(--text);
            font-family: 'Roboto', sans-serif;
            transition: all 0.3s;
        }
        
        input:focus, select:focus, textarea:focus, button:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(187, 134, 252, 0.3);
        }
        
        button {
            cursor: pointer;
            background: var(--accent);
            color: #121212;
            font-weight: 500;
            border: none;
            padding: 10px 20px;
        }
        
        button:hover {
            background: #9a67e0;
        }
        
        label {
            color: var(--text);
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .data-card {
            background: rgba(30, 30, 50, 0.6);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 3px solid var(--primary);
            transition: transform 0.3s ease;
        }
        
        .data-card:hover {
            transform: translateX(5px);
        }
        
        .data-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .stat-card {
            background: rgba(30, 30, 50, 0.6);
            border-radius: 10px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .stat-card:nth-child(1) {
            border-top: 3px solid var(--primary);
        }
        
        .stat-card:nth-child(2) {
            border-top: 3px solid var(--secondary);
        }
        
        .stat-card:nth-child(3) {
            border-top: 3px solid var(--success);
        }
        
        .stat-card:nth-child(4) {
            border-top: 3px solid var(--info);
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        
        .stat-card h3 {
            font-size: 2rem;
            margin: 10px 0;
        }
        
        .stat-card p {
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.9rem;
        }
        
        footer {
            margin-left: var(--sidebar-width);
            padding: 20px 30px;
            text-align: center;
            font-size: 0.9rem;
            color: rgba(255, 255, 255, 0.5);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            transition: margin 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        @media (max-width: 992px) {
            .data-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        
        @media (max-width: 768px) {
            .sidebar {
                transform: translateX(-100%);
            }
            
            .sidebar.active {
                transform: translateX(0);
            }
            
            .mobile-toggle {
                display: flex;
            }
            
            .main-content, footer {
                margin-left: 0;
                padding: 20px 15px;
            }
            
            .data-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div id="particles-js"></div>
    
    <button class="mobile-toggle" id="sidebarToggle" aria-label="Toggle sidebar">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="3" y1="12" x2="21" y2="12"></line>
            <line x1="3" y1="6" x2="21" y2="6"></line>
            <line x1="3" y1="18" x2="21" y2="18"></line>
        </svg>
    </button>

    <div class="sidebar" id="sidebar">
        <div class="sidebar-header">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2"></path>
            </svg>
            <h3>Healthcare AI</h3>
        </div>
        <div class="nav-links">
            <a href="/" class="active" aria-label="Go to Home page">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                    <polyline points="9 22 9 12 15 12 15 22"></polyline>
                </svg>
                Home
            </a>
            <a href="/dataset" aria-label="View loaded dataset">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <ellipse cx="12" cy="5" rx="9" ry="3"></ellipse>
                    <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path>
                    <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path>
                </svg>
                Dataset
            </a>
            <a href="/model_visualisations" aria-label="Visualize model performance">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                </svg>
                Model Insights
            </a>
            <a href="/patient_visualisations" aria-label="Visualize patient results">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                    <circle cx="12" cy="7" r="4"></circle>
                </svg>
                Patient Analytics
            </a>
        </div>
    </div>
    
    <div class="main-content">
        <div class="page-header">
            <h1>{{ title }}</h1>
        </div>
        
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="flash-message {{ category }}" role="alert">
                        {{ message }}
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        <div class="glass-container">
            {{ content | safe }}
        </div>
    </div>
    
    <footer>
        © Healthcare AI Platform 2025 | Advanced Patient Analytics & Insights
    </footer>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/particles.js/2.0.0/particles.min.js"></script>
    <script src="{{ url_for('static', filename='js/particles-config.js') }}"></script>
    
    <script>
        // Mobile sidebar toggle
        document.getElementById('sidebarToggle').addEventListener('click', function() {
            document.getElementById('sidebar').classList.toggle('active');
        });
        
        // Set active nav item based on current page
        document.addEventListener('DOMContentLoaded', function() {
            const currentLocation = window.location.pathname;
            const navLinks = document.querySelectorAll('.sidebar a');
            
            navLinks.forEach(link => {
                if (link.getAttribute('href') === currentLocation) {
                    link.classList.add('active');
                } else {
                    link.classList.remove('active');
                }
            });
        });
    </script>
</body>
</html>
"""

# Predict page template (fully hardcoded)
PREDICT_HTML = """
<h1 style="margin-bottom: 20px;">Predict Heart Disease</h1>
<p style="color: #ef5350; margin-bottom: 20px;">
    <strong>Disclaimer:</strong> This tool is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.
</p>
<form id="prediction-form" style="margin-bottom: 20px;">
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
        <!-- Age -->
        <div style="display: flex; flex-direction: column;">
            <label for="Age" style="margin-bottom: 5px; font-weight: 500;">Age</label>
            <input type="number" id="Age" name="Age" step="any" value="63.0"
                   required aria-label="Patient age in years"
                   style="padding: 8px; background: rgba(255, 255, 255, 0.1); color: #F5F5F5; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px;">
        </div>
        <!-- Sex -->
        <div style="display: flex; flex-direction: column;">
            <label for="Sex" style="margin-bottom: 5px; font-weight: 500;">Sex</label>
            <select id="Sex" name="Sex" required aria-label="Patient sex"
                    style="padding: 8px; background: rgba(255, 255, 255, 0.1); color: #F5F5F5; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px;">
                <option value="0">Female</option>
                <option value="1" selected>Male</option>
            </select>
        </div>
        <!-- CP (Chest Pain Type) -->
        <div style="display: flex; flex-direction: column;">
            <label for="CP" style="margin-bottom: 5px; font-weight: 500;">Chest Pain Type</label>
            <select id="CP" name="CP" required aria-label="Chest pain type"
                    style="padding: 8px; background: rgba(255, 255, 255, 0.1); color: #F5F5F5; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px;">
                <option value="0">Asymptomatic</option>
                <option value="1">Typical Angina</option>
                <option value="2">Atypical Angina</option>
                <option value="3" selected>Non-Anginal Pain</option>
            </select>
        </div>
        <!-- Trestbps -->
        <div style="display: flex; flex-direction: column;">
            <label for="Trestbps" style="margin-bottom: 5px; font-weight: 500;">Resting BP</label>
            <input type="number" id="Trestbps" name="Trestbps" step="any" value="145.0"
                   required aria-label="Resting blood pressure in mm Hg"
                   style="padding: 8px; background: rgba(255, 255, 255, 0.1); color: #F5F5F5; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px;">
        </div>
        <!-- Chol -->
        <div style="display: flex; flex-direction: column;">
            <label for="Chol" style="margin-bottom: 5px; font-weight: 500;">Cholesterol</label>
            <input type="number" id="Chol" name="Chol" step="any" value="233.0"
                   required aria-label="Serum cholesterol in mg/dl"
                   style="padding: 8px; background: rgba(255, 255, 255, 0.1); color: #F5F5F5; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px;">
        </div>
        <!-- Fbs -->
        <div style="display: flex; flex-direction: column;">
            <label for="Fbs" style="margin-bottom: 5px; font-weight: 500;">Fasting Blood Sugar</label>
            <select id="Fbs" name="Fbs" required aria-label="Fasting blood sugar level"
                    style="padding: 8px; background: rgba(255, 255, 255, 0.1); color: #F5F5F5; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px;">
                <option value="0">≤ 120 mg/dl</option>
                <option value="1" selected>> 120 mg/dl</option>
            </select>
        </div>
        <!-- Restecg -->
        <div style="display: flex; flex-direction: column;">
            <label for="Restecg" style="margin-bottom: 5px; font-weight: 500;">Resting ECG</label>
            <select id="Restecg" name="Restecg" required aria-label="Resting electrocardiographic results"
                    style="padding: 8px; background: rgba(255, 255, 255, 0.1); color: #F5F5F5; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px;">
                <option value="0" selected>Normal</option>
                <option value="1">ST-T Wave Abnormality</option>
                <option value="2">Left Ventricular Hypertrophy</option>
            </select>
        </div>
        <!-- Thalach -->
        <div style="display: flex; flex-direction: column;">
            <label for="Thalach" style="margin-bottom: 5px; font-weight: 500;">Max Heart Rate</label>
            <input type="number" id="Thalach" name="Thalach" step="any" value="150.0"
                   required aria-label="Maximum heart rate achieved"
                   style="padding: 8px; background: rgba(255, 255, 255, 0.1); color: #F5F5F5; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px;">
        </div>
        <!-- Exang -->
        <div style="display: flex; flex-direction: column;">
            <label for="Exang" style="margin-bottom: 5px; font-weight: 500;">Exercise Angina</label>
            <select id="Exang" name="Exang" required aria-label="Exercise-induced angina"
                    style="padding: 8px; background: rgba(255, 255, 255, 0.1); color: #F5F5F5; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px;">
                <option value="0" selected>No</option>
                <option value="1">Yes</option>
            </select>
        </div>
        <!-- Oldpeak -->
        <div style="display: flex; flex-direction: column;">
            <label for="Oldpeak" style="margin-bottom: 5px; font-weight: 500;">ST Depression</label>
            <input type="number" id="Oldpeak" name="Oldpeak" step="any" value="2.3"
                   required aria-label="ST depression induced by exercise"
                   style="padding: 8px; background: rgba(255, 255, 255, 0.1); color: #F5F5F5; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px;">
        </div>
        <!-- Slope -->
        <div style="display: flex; flex-direction: column;">
            <label for="Slope" style="margin-bottom: 5px; font-weight: 500;">Slope</label>
            <select id="Slope" name="Slope" required aria-label="Slope of the peak exercise ST segment"
                    style="padding: 8px; background: rgba(255, 255, 255, 0.1); color: #F5F5F5; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px;">
                <option value="0" selected>Unsloping</option>
                <option value="1">Flat</option>
                <option value="2">Downsloping</option>
            </select>
        </div>
        <!-- CA -->
        <div style="display: flex; flex-direction: column;">
            <label for="CA" style="margin-bottom: 5px; font-weight: 500;">Major Vessels</label>
            <select id="CA" name="CA" required aria-label="Number of major vessels colored by fluoroscopy"
                    style="padding: 8px; background: rgba(255, 255, 255, 0.1); color: #F5F5F5; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px;">
                <option value="0" selected>0 Vessels</option>
                <option value="1">1 Vessel</option>
                <option value="2">2 Vessels</option>
                <option value="3">3 Vessels</option>
                <option value="4">4 Vessels</option>
            </select>
        </div>
        <!-- Thal -->
        <div style="display: flex; flex-direction: column;">
            <label for="Thal" style="margin-bottom: 5px; font-weight: 500;">Thalassemia</label>
            <select id="Thal" name="Thal" required aria-label="Thalassemia condition"
                    style="padding: 8px; background: rgba(255, 255, 255, 0.1); color: #F5F5F5; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px;">
                <option value="0">Unknown</option>
                <option value="1" selected>Normal</option>
                <option value="2">Fixed Defect</option>
                <option value="3">Reversible Defect</option>
            </select>
        </div>
    </div>
    <div id="error-message" style="color: #ef5350; margin-top: 10px; display: none;" role="alert"></div>
    <button type="submit" style="margin-top: 20px; padding: 10px 20px; background: #BB86FC; color: #121212; border: none; border-radius: 8px; cursor: pointer;">
        Predict
    </button>
</form>
<div id="loading" style="display: none; position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(0, 0, 0, 0.8); color: #BB86FC; padding: 10px 20px; border-radius: 8px;">
    Predicting...
</div>
<div id="prediction-result" style="margin-top: 20px;">
    <h3 style="margin-bottom: 20px;">Prediction Result</h3>
    <p>Submit the form to see the prediction result.</p>
</div>
<script>
    function validateForm() {
        const inputs = document.querySelectorAll('input[type="number"]');
        const errorDiv = document.getElementById('error-message');
        errorDiv.style.display = 'none';
        errorDiv.innerText = '';
        for (let input of inputs) {
            const value = input.value.trim();
            if (value === '' || isNaN(value)) {
                errorDiv.innerText = `Please enter a valid number for ${input.name}.`;
                errorDiv.style.display = 'block';
                return false;
            }
        }
        return true;
    }

    async function handleSubmit(event) {
        event.preventDefault();
        if (!validateForm()) return;

        const loadingDiv = document.getElementById('loading');
        loadingDiv.style.display = 'block';

        const formData = new FormData(document.getElementById('prediction-form'));
        try {
            const response = await fetch('/', {
                method: 'POST',
                body: formData
            });
            const html = await response.text();
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, 'text/html');
            const newResult = doc.querySelector('#prediction-result').innerHTML;
            document.getElementById('prediction-result').innerHTML = newResult;

            // Display flash messages
            const flashMessages = doc.querySelectorAll('.flash-message');
            const mainContent = document.querySelector('.main-content');
            const existingFlash = document.querySelectorAll('.flash-message');
            existingFlash.forEach(msg => msg.remove());
            flashMessages.forEach(msg => mainContent.insertBefore(msg, mainContent.firstChild));
        } catch (error) {
            const errorDiv = document.getElementById('error-message');
            errorDiv.innerText = 'An error occurred while predicting. Please try again.';
            errorDiv.style.display = 'block';
        } finally {
            loadingDiv.style.display = 'none';
        }
    }

    document.getElementById('prediction-form').onsubmit = handleSubmit;
</script>
"""

# Dataset page template
DATASET_HTML = """
<h1 style="margin-bottom: 20px;">Loaded Dataset</h1>
<div style="overflow-x: auto;">
    <table style="width: 100%; border-collapse: collapse; border-radius: 8px; overflow: hidden;">
        <thead style="background: rgba(187, 134, 252, 0.3);">
            <tr>
                {% for column in columns %}
                <th style="padding: 12px; text-align: left; border-bottom: 1px solid rgba(255, 255, 255, 0.2);">{{ column }}</th>
                {% endfor %}
            </tr>
        </thead>
        <tbody>
            {% for _, row in dataset.iterrows() %}
            {% if loop.index <= 100 %}
            <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.1);">
                {% for column in columns %}
                <td style="padding: 10px; text-align: left;">{{ row[column] }}</td>
                {% endfor %}
            </tr>
            {% endif %}
            {% endfor %}
        </tbody>
    </table>
    {% if dataset|length > 100 %}
    <p style="margin-top: 15px; text-align: center; color: #BB86FC;">
        Showing first 100 rows of {{ dataset|length }} total rows.
    </p>
    {% endif %}
</div>
"""

# Prediction result template (for POST response)
PREDICTION_RESULT_HTML = """
<h3 style="margin-bottom: 20px;">Prediction Result</h3>
<div style="background: {{ '#ef5350' if prediction_class == 'danger' else '#4caf50' }}; padding: 15px; border-radius: 8px;" role="alert">
    <strong>{{ prediction_text }}</strong>
    <p>Probability: {{ probability }}%</p>
</div>
"""

# Visualization creation
def create_visualization(viz_type, data, figsize=(10, 8), save_path=None):
    global fig
    try:
        fig, ax = plt.subplots(figsize=figsize)
        if viz_type.lower() == "roc_curve":
            if not all(k in data for k in ['X_test_scaled', 'y_test', 'model']):
                return None
            model = data['model']
            X_test_scaled = data['X_test_scaled']
            y_test = data['y_test']
            if len(model.classes_) != 2:
                return None
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], 'k--', lw=2)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic')
            ax.legend(loc="lower right")
        elif viz_type.lower() == "confusion_matrix":
            if not all(k in data for k in ['X_test_scaled', 'y_test', 'model']):
                return None
            model = data['model']
            X_test_scaled = data['X_test_scaled']
            y_test = data['y_test']
            y_pred = model.predict(X_test_scaled)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted Labels')
            ax.set_ylabel('True Labels')
            ax.set_title('Confusion Matrix')
        elif viz_type.lower() == "feature_importance":
            if 'feature_importances' not in data or not data['feature_importances']:
                return None
            feature_importances = data['feature_importances']
            sorted_features = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))
            if len(sorted_features) > 20:
                top_features = dict(list(sorted_features.items())[:20])
                sorted_features = top_features
                ax.set_title('Top 20 Features by Importance')
            else:
                ax.set_title('Feature Importance')
            bars = ax.barh(list(sorted_features.keys()), list(sorted_features.values()))
            ax.set_xlabel('Importance')
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{width:.3f}', ha='left', va='center')
        elif viz_type.lower() == "correlation_matrix":
            if 'features' not in data or not isinstance(data['features'], pd.DataFrame):
                return None
            features = data['features']
            corr_matrix = features.corr()
            if corr_matrix.shape[0] > 20:
                mask = np.abs(corr_matrix) < 0.3
                np.fill_diagonal(mask, False)
                sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                            vmin=-1, vmax=1, ax=ax, mask=mask)
                ax.set_title('Significant Feature Correlations (|r| >= 0.3)')
            else:
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                            vmin=-1, vmax=1, ax=ax, fmt='.2f')
                ax.set_title('Feature Correlation Matrix')
        elif viz_type.lower() == "patient_distribution":
            if not all(k in data for k in ['X_test_scaled', 'y_test', 'model']):
                return None
            model = data['model']
            X_test_scaled = data['X_test_scaled']
            y_test = data['y_test']
            y_pred = model.predict(X_test_scaled)
            actual_counts = pd.Series(y_test).value_counts().sort_index()
            predicted_counts = pd.Series(y_pred).value_counts().sort_index()
            classes = np.sort(np.unique(np.concatenate([y_test, y_pred])))
            if len(classes) == 2:
                labels = ['No Disease', 'Disease']
            else:
                labels = [f'Class {c}' for c in classes]
            x = np.arange(len(labels))
            width = 0.35
            actual_values = [actual_counts.get(c, 0) for c in classes]
            predicted_values = [predicted_counts.get(c, 0) for c in classes]
            ax.bar(x - width / 2, actual_values, width, label='Actual', color='skyblue')
            ax.bar(x + width / 2, predicted_values, width, label='Predicted', color='salmon')
            ax.set_ylabel('Number of Patients')
            ax.set_title('Patient Distribution: Actual vs Predicted')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
            for i, v in enumerate(actual_values):
                ax.text(i - width / 2, v + 0.5, str(v), ha='center')
            for i, v in enumerate(predicted_values):
                ax.text(i + width / 2, v + 0.5, str(v), ha='center')
        elif viz_type.lower() == "precision_recall_curve":
            if not all(k in data for k in ['X_test_scaled', 'y_test', 'model']):
                return None
            model = data['model']
            X_test_scaled = data['X_test_scaled']
            y_test = data['y_test']
            if len(model.classes_) != 2:
                return None
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            ax.plot(recall, precision, marker='.', label='Precision-Recall curve')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.grid(True)
            ap = precision_score(y_test, model.predict(X_test_scaled))
            ax.text(0.05, 0.05, f'Average Precision: {ap:.3f}', transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))
        elif viz_type.lower() == "model_metrics":
            if not all(k in data for k in ['X_test_scaled', 'y_test', 'model']):
                return None
            model = data['model']
            X_test_scaled = data['X_test_scaled']
            y_test = data['y_test']
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            if len(model.classes_) == 2:
                try:
                    y_prob = model.predict_proba(X_test_scaled)[:, 1]
                    auc_score = roc_auc_score(y_test, y_prob)
                except:
                    auc_score = 0
            else:
                auc_score = 0
            metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'AUC-ROC']
            values = [accuracy, f1, precision, recall, auc_score]
            bars = ax.bar(metrics, values, color=['#4CAF50', '#2196F3', '#FFC107', '#9C27B0', '#FF5722'])
            ax.set_ylim([0, 1.1])
            ax.set_title('Model Performance Metrics')
            ax.set_ylabel('Score')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                        f'{height:.3f}', ha='center', va='bottom')
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        else:
            plt.close(fig)
            return None
        fig.tight_layout()
        if save_path:
            try:
                fig.savefig(save_path, bbox_inches='tight', dpi=300)
            except Exception:
                pass
        return fig
    except Exception:
        if 'fig' in locals():
            plt.close(fig)
        return None

# Class entry
class HealthcareAI:
    def __init__(self):
        self.dataset = None
        self.features = None
        self.target = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.model_accuracies = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.original_columns = []
        self.dummy_columns = []
        self.load_data()
        if self.dataset is not None:
            self.preprocess_data()
            self.train_models()

    def load_data(self):
        try:
            self.dataset = pd.read_csv("heart.csv")
            if self.dataset.empty:
                return False
            return True
        except Exception:
            return False

    def preprocess_data(self, test_size=0.2):
        if self.dataset is None:
            return False
        target_column = 'Target'
        if target_column not in self.dataset.columns:
            return False
        for col in self.dataset.columns:
            if self.dataset[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(self.dataset[col]):
                    self.dataset[col].fillna(self.dataset[col].median(), inplace=True)
                else:
                    self.dataset[col].fillna(self.dataset[col].mode()[0], inplace=True)
        for cat_col in CATEGORICAL_OPTIONS.keys():
            if cat_col in self.dataset.columns:
                valid_options = set(CATEGORICAL_OPTIONS[cat_col])
                actual_values = set(self.dataset[cat_col].astype(str))
                if not actual_values.issubset(valid_options):
                    self.dataset[cat_col] = self.dataset[cat_col].astype(str).apply(
                        lambda x: x if x in valid_options else valid_options.pop()
                    )
        self.target = self.dataset[target_column]
        self.features = self.dataset.drop(target_column, axis=1)
        self.original_columns = self.features.columns.tolist()
        categorical_cols = self.features.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            self.features = pd.get_dummies(self.features, columns=categorical_cols)
        self.dummy_columns = self.features.columns.tolist()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target, test_size=test_size, random_state=42)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        return True

    def train_models(self):
        if not hasattr(self, 'X_train_scaled') or self.X_train_scaled is None:
            return False
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
        }
        self.model_accuracies = {}
        for name, model in models.items():
            try:
                model.fit(self.X_train_scaled, self.y_train)
                self.models[name] = model
                y_pred = model.predict(self.X_test_scaled)
                test_accuracy = accuracy_score(self.y_test, y_pred)
                self.model_accuracies[name] = {"test_accuracy": float(test_accuracy * 100)}
            except Exception:
                self.model_accuracies[name] = {"test_accuracy": 0.0}
        if not self.model_accuracies:
            return False
        self.best_model_name = max(self.model_accuracies, key=lambda x: self.model_accuracies[x]["test_accuracy"])
        self.best_model = self.models[self.best_model_name]
        return True

    def predict(self, patient_data, threshold=0.5):
        if self.best_model is None:
            return None
        try:
            input_df = pd.DataFrame([patient_data], columns=self.original_columns)
            categorical_cols = input_df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                input_df = pd.get_dummies(input_df, columns=categorical_cols)
            for col in self.dummy_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[self.dummy_columns]
            scaled_data = self.scaler.transform(input_df)
            probabilities = self.best_model.predict_proba(scaled_data)[0]
            probability = probabilities[1]
            prediction = 1 if probability >= threshold else 0
            return {
                "probability": probability,
                "prediction": prediction
            }
        except Exception:
            return None

# Initialize AI system
ai_system = HealthcareAI()

@app.route('/', methods=['GET', 'POST'])
def predict():
    expected_features = ['Age', 'Sex', 'CP', 'Trestbps', 'Chol', 'Fbs', 'Restecg', 'Thalach', 'Exang', 'Oldpeak',
                         'Slope', 'CA', 'Thal']
    content = PREDICT_HTML
    if ai_system.dataset is None:
        flash('Dataset not loaded. Ensure heart.csv is in the correct directory.', 'error')
    elif not hasattr(ai_system, 'features') or ai_system.features is None:
        flash('Model not trained. Check dataset and preprocessing.', 'error')
    elif not all(f in ai_system.original_columns for f in expected_features):
        flash('Dataset columns do not match expected features. Check heart.csv.', 'error')
    elif request.method == 'POST':
        patient_data = {}
        for field, value in request.form.items():
            try:
                if field in ai_system.original_columns:
                    if field in CATEGORICAL_OPTIONS:
                        if value not in CATEGORICAL_OPTIONS[field]:
                            flash(f"Invalid value for {field}. Select a valid option.", 'error')
                            break
                        patient_data[field] = int(value)
                    else:
                        patient_data[field] = float(value)
            except ValueError:
                flash(f"Invalid value for {field}. Please enter a valid number.", 'error')
                break
        else:  # No break occurred, process prediction
            result = ai_system.predict(patient_data)
            if result:
                prediction_text = 'Disease Detected' if result['prediction'] == 1 else 'No Disease Detected'
                probability = round(result['probability'] * 100, 2)
                prediction_class = 'danger' if result['prediction'] == 1 else 'success'
                result_html = render_template_string(PREDICTION_RESULT_HTML, prediction_text=prediction_text,
                                                     probability=probability, prediction_class=prediction_class)
                content = PREDICT_HTML.replace(
                    '<div id="prediction-result" style="margin-top: 20px;">\n    <h3 style="margin-bottom: 20px;">Prediction Result</h3>\n    <p>Submit the form to see the prediction result.</p>\n</div>',
                    f'<div id="prediction-result" style="margin-top: 20px;">\n{result_html}\n</div>'
                )
            else:
                flash('Prediction failed. Ensure all required features are provided correctly.', 'error')
    try:
        response = make_response(render_template_string(BASE_HTML, title="Predict Heart Disease", content=content))
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
    except Exception as e:
        flash(f"Error rendering page: {str(e)}", 'error')
        response = make_response(render_template_string(BASE_HTML, title="Predict Heart Disease", content=PREDICT_HTML))
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

@app.route('/dataset')
def show_dataset():
    if ai_system.dataset is None:
        flash('Dataset not loaded. Ensure heart.csv is in the correct directory.', 'error')
        content = "<p>No dataset available. Please ensure the dataset is loaded.</p>"
    else:
        try:
            # Limited to first 100 rows for performance
            content = render_template_string(DATASET_HTML,
                                             dataset=ai_system.dataset.head(100),
                                             columns=ai_system.dataset.columns)
        except Exception as e:
            flash(f"Error displaying dataset: {str(e)}", 'error')
            content = "<p>Error displaying dataset.</p>"

    try:
        response = make_response(render_template_string(BASE_HTML, title="Loaded Dataset", content=content))
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
    except Exception as e:
        flash(f"Error rendering page: {str(e)}", 'error')
        response = make_response(
            render_template_string(BASE_HTML, title="Loaded Dataset", content="<p>Error loading page.</p>"))
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

@app.route('/model_visualisations')
def model_visualisations():
    images_dir = os.path.join(app.root_path, 'static', 'images')
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
    os.makedirs(images_dir, exist_ok=True)

    if (ai_system.best_model is None or ai_system.X_test_scaled is None or
            ai_system.y_test is None or ai_system.features is None):
        flash('Cannot generate visualizations: Model or data not available.', 'error')
        content = "<p>No visualizations available. Please ensure the model is trained and data is loaded.</p>"
        response = make_response(render_template_string(BASE_HTML, title="Model Visualisations", content=content))
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    common_data = {
        'X_test_scaled': ai_system.X_test_scaled,
        'y_test': ai_system.y_test,
        'model': ai_system.best_model
    }

    visualizations = {
        'model_metrics': None,
        'roc_curve': None,
        'precision_recall_curve': None,
        'confusion_matrix': None
    }

    # Model Metrics
    fig = create_visualization('model_metrics', common_data, figsize=(10, 6),
                               save_path=os.path.join(images_dir, 'model_metrics.png'))
    if fig:
        visualizations['model_metrics'] = '/static/images/model_metrics.png'
        plt.close(fig)

    # ROC Curve
    fig = create_visualization('roc_curve', common_data, figsize=(8, 6),
                               save_path=os.path.join(images_dir, 'roc_curve.png'))
    if fig:
        visualizations['roc_curve'] = '/static/images/roc_curve.png'
        plt.close(fig)

    # Precision-Recall Curve
    fig = create_visualization('precision_recall_curve', common_data, figsize=(8, 6),
                               save_path=os.path.join(images_dir, 'precision_recall_curve.png'))
    if fig:
        visualizations['precision_recall_curve'] = '/static/images/precision_recall_curve.png'
        plt.close(fig)

    # Confusion Matrix
    fig = create_visualization('confusion_matrix', common_data, figsize=(6, 6),
                               save_path=os.path.join(images_dir, 'confusion_matrix.png'))
    if fig:
        visualizations['confusion_matrix'] = '/static/images/confusion_matrix.png'
        plt.close(fig)

    try:
        # Use only the Model Visualisations section from the VISUALISE_HTML template
        model_visualisations_html = """
        <h1 style="margin-bottom: 20px;">Model Performance Visualizations</h1>

        <!-- Model Visualisations Section -->
        <div style="background: rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 20px; margin-bottom: 30px;">
            <div style="display: flex; flex-direction: column; gap: 20px;">
                {% if model_metrics %}
                <div>
                    <h3>Model Performance Metrics</h3>
                    <img src="{{ model_metrics }}" alt="Model Performance Metrics" style="max-width: 100%; border-radius: 8px;">
                </div>
                {% endif %}
                {% if roc_curve %}
                <div>
                    <h3>ROC Curve</h3>
                    <img src="{{ roc_curve }}" alt="ROC Curve" style="max-width: 100%; border-radius: 8px;">
                </div>
                {% endif %}
                {% if precision_recall_curve %}
                <div>
                    <h3>Precision-Recall Curve</h3>
                    <img src="{{ precision_recall_curve }}" alt="Precision-Recall Curve" style="max-width: 100%; border-radius: 8px;">
                </div>
                {% endif %}
                {% if confusion_matrix %}
                <div>
                    <h3>Confusion Matrix</h3>
                    <img src="{{ confusion_matrix }}" alt="Confusion Matrix" style="max-width: 100%; border-radius: 8px;">
                </div>
                {% endif %}
            </div>
        </div>
        """

        content = render_template_string(model_visualisations_html, **visualizations)
        response = make_response(render_template_string(BASE_HTML, title="Model Visualisations", content=content))
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
    except Exception as e:
        flash(f"Error rendering visualisation page: {str(e)}", 'error')
        response = make_response(render_template_string(BASE_HTML, title="Model Visualisations",
                                                        content="<p>Error loading visualisations.</p>"))
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

@app.route('/patient_visualisations')
def patient_visualisations():
    images_dir = os.path.join(app.root_path, 'static', 'images')
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
    os.makedirs(images_dir, exist_ok=True)

    if (ai_system.best_model is None or ai_system.X_test_scaled is None or
            ai_system.y_test is None or ai_system.features is None):
        flash('Cannot generate visualizations: Model or data not available.', 'error')
        content = "<p>No visualizations available. Please ensure the model is trained and data is loaded.</p>"
        response = make_response(
            render_template_string(BASE_HTML, title="Patient Result Visualisations", content=content))
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    common_data = {
        'X_test_scaled': ai_system.X_test_scaled,
        'y_test': ai_system.y_test,
        'model': ai_system.best_model
    }

    visualizations = {
        'feature_importance': None,
        'correlation_matrix': None,
        'patient_distribution': None
    }

    # Feature Importance
    feature_importances = {}
    if ai_system.best_model_name == 'Logistic Regression':
        coef = np.abs(ai_system.best_model.coef_[0])
        feature_importances = dict(zip(ai_system.dummy_columns, coef))
    elif hasattr(ai_system.best_model, 'feature_importances_'):
        feature_importances = dict(zip(ai_system.dummy_columns, ai_system.best_model.feature_importances_))

    if feature_importances:
        fig = create_visualization('feature_importance', {'feature_importances': feature_importances},
                                   figsize=(10, 8), save_path=os.path.join(images_dir, 'feature_importance.png'))
        if fig:
            visualizations['feature_importance'] = '/static/images/feature_importance.png'
            plt.close(fig)

    # Correlation Matrix
    fig = create_visualization('correlation_matrix', {'features': ai_system.dataset[ai_system.original_columns]},
                               figsize=(10, 8), save_path=os.path.join(images_dir, 'correlation_matrix.png'))
    if fig:
        visualizations['correlation_matrix'] = '/static/images/correlation_matrix.png'
        plt.close(fig)

    # Patient Distribution
    fig = create_visualization('patient_distribution', common_data, figsize=(8, 6),
                               save_path=os.path.join(images_dir, 'patient_distribution.png'))
    if fig:
        visualizations['patient_distribution'] = '/static/images/patient_distribution.png'
        plt.close(fig)

    try:
        patient_visualisations_html = """
        <h1 style="margin-bottom: 20px;">Patient Result Visualizations</h1>

        <!-- Patient Result Visualisations Section -->
        <div style="background: rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 20px;">
            <div style="display: flex; flex-direction: column; gap: 20px;">
                {% if feature_importance %}
                <div>
                    <h3>Feature Importance</h3>
                    <img src="{{ feature_importance }}" alt="Feature Importance" style="max-width: 100%; border-radius: 8px;">
                </div>
                {% endif %}
                {% if correlation_matrix %}
                <div>
                    <h3>Correlation Matrix</h3>
                    <img src="{{ correlation_matrix }}" alt="Correlation Matrix" style="max-width: 100%; border-radius: 8px;">
                </div>
                {% endif %}
                {% if patient_distribution %}
                <div>
                    <h3>Patient Distribution</h3>
                    <img src="{{ patient_distribution }}" alt="Patient Distribution" style="max-width: 100%; border-radius: 8px;">
                </div>
                {% endif %}
            </div>
        </div>
        """

        content = render_template_string(patient_visualisations_html, **visualizations)
        response = make_response(
            render_template_string(BASE_HTML, title="Patient Result Visualisations", content=content))
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
    except Exception as e:
        flash(f"Error rendering visualisation page: {str(e)}", 'error')
        response = make_response(render_template_string(BASE_HTML, title="Patient Result Visualisations",
                                                        content="<p>Error loading visualisations.</p>"))
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

@app.route('/images/<filename>')
def serve_image(filename):
    try:
        return send_from_directory(os.path.join(app.root_path, 'static', 'images'), filename)
    except Exception as e:
        flash(f"Error serving image: {str(e)}", 'error')
        return '', 404

@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)