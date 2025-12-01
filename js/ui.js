/**
 * UI Utilities Module
 * Handles UI interactions and visual feedback
 */

/**
 * Show a toast notification
 */
export function showToast(message, type = 'info', duration = 3000) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

/**
 * Format time in HH:MM:SS format
 */
export function formatTime(seconds) {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Format percentage
 */
export function formatPercent(value) {
    return (value * 100).toFixed(2) + '%';
}

/**
 * Read file as data URL
 */
export function readFileAsDataURL(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

/**
 * Check if file is an image
 */
export function isImageFile(file) {
    return file.type.startsWith('image/');
}

/**
 * Create a chart instance
 */
export function createChart(canvasId, type, label, borderColor, backgroundColor) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: label,
                data: [],
                borderColor: borderColor,
                backgroundColor: backgroundColor,
                borderWidth: 2,
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Epoch'
                    }
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: label
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            animation: {
                duration: 0
            }
        }
    });
}

/**
 * Create probability bar chart
 */
export function createProbabilityChart(canvasId) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Probability',
                data: [],
                backgroundColor: [],
                borderColor: [],
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    beginAtZero: true,
                    max: 1,
                    title: {
                        display: true,
                        text: 'Probability'
                    },
                    ticks: {
                        callback: function(value) {
                            return (value * 100) + '%';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return (context.raw * 100).toFixed(2) + '%';
                        }
                    }
                }
            }
        }
    });
}

/**
 * Update chart data
 */
export function updateChart(chart, label, value) {
    chart.data.labels.push(label);
    chart.data.datasets[0].data.push(value);
    chart.update('none');
}

/**
 * Reset chart data
 */
export function resetChart(chart) {
    chart.data.labels = [];
    chart.data.datasets[0].data = [];
    chart.update('none');
}

/**
 * Update probability chart with predictions
 */
export function updateProbabilityChart(chart, predictions) {
    const maxProb = Math.max(...predictions.map(p => p.probability));

    chart.data.labels = predictions.map(p => p.className);
    chart.data.datasets[0].data = predictions.map(p => p.probability);
    chart.data.datasets[0].backgroundColor = predictions.map(p =>
        p.probability === maxProb ? 'rgba(40, 167, 69, 0.8)' : 'rgba(74, 144, 217, 0.6)'
    );
    chart.data.datasets[0].borderColor = predictions.map(p =>
        p.probability === maxProb ? 'rgb(40, 167, 69)' : 'rgb(74, 144, 217)'
    );
    chart.update('none');
}

/**
 * Debounce function
 */
export function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Create element with class and content
 */
export function createElement(tag, className, content) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (content) element.textContent = content;
    return element;
}

/**
 * Enable/disable form elements
 */
export function setFormEnabled(selector, enabled) {
    const elements = document.querySelectorAll(selector);
    elements.forEach(el => {
        el.disabled = !enabled;
    });
}

/**
 * Memory management utility
 */
export function cleanupMemory() {
    if (typeof tf !== 'undefined') {
        tf.disposeVariables();
    }
}
