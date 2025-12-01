/**
 * Main Application Module
 * TensorFlow.js Image Classifier - Pure Frontend Application
 */

import { storage } from './storage.js';
import { modelManager } from './model.js';
import {
    showToast,
    formatTime,
    formatPercent,
    readFileAsDataURL,
    isImageFile,
    createChart,
    createProbabilityChart,
    updateChart,
    resetChart,
    updateProbabilityChart,
    debounce
} from './ui.js';

// ==================== Application State ====================
const state = {
    classes: [],
    selectedClassId: null,
    isTraining: false,
    isPaused: false,
    trainingStartTime: null,
    lossChart: null,
    accuracyChart: null,
    probabilityChart: null,
    timerInterval: null
};

// ==================== DOM Elements ====================
const elements = {
    // Class management
    newClassName: document.getElementById('new-class-name'),
    addClassBtn: document.getElementById('add-class-btn'),
    classesContainer: document.getElementById('classes-container'),
    dropZone: document.getElementById('drop-zone'),
    fileInput: document.getElementById('file-input'),
    classSelector: document.getElementById('class-selector'),
    targetClass: document.getElementById('target-class'),

    // Training parameters
    learningRate: document.getElementById('learning-rate'),
    lrValue: document.getElementById('lr-value'),
    batchSize: document.getElementById('batch-size'),
    epochs: document.getElementById('epochs'),
    epochsValue: document.getElementById('epochs-value'),
    modelArchitecture: document.getElementById('model-architecture'),

    // Training control
    startTrainingBtn: document.getElementById('start-training-btn'),
    pauseTrainingBtn: document.getElementById('pause-training-btn'),
    stopTrainingBtn: document.getElementById('stop-training-btn'),
    trainingStatus: document.getElementById('training-status'),
    currentEpoch: document.getElementById('current-epoch'),
    batchProgress: document.getElementById('batch-progress'),
    elapsedTime: document.getElementById('elapsed-time'),
    remainingTime: document.getElementById('remaining-time'),
    trainingProgress: document.getElementById('training-progress'),

    // Charts
    lossChart: document.getElementById('loss-chart'),
    accuracyChart: document.getElementById('accuracy-chart'),

    // Validation
    testDropZone: document.getElementById('test-drop-zone'),
    testFileInput: document.getElementById('test-file-input'),
    predictionResults: document.getElementById('prediction-results'),
    testImagePreview: document.getElementById('test-image-preview'),
    predictedClass: document.getElementById('predicted-class'),
    probabilityChart: document.getElementById('probability-chart'),

    // Stats
    totalImages: document.getElementById('total-images'),
    numClasses: document.getElementById('num-classes'),
    finalAccuracy: document.getElementById('final-accuracy'),
    modelStatus: document.getElementById('model-status'),

    // Model actions
    saveModelBtn: document.getElementById('save-model-btn'),
    loadModelBtn: document.getElementById('load-model-btn'),
    clearDataBtn: document.getElementById('clear-data-btn')
};

// ==================== Initialization ====================
async function init() {
    try {
        // Initialize storage
        await storage.init();

        // Initialize charts
        initCharts();

        // Load existing data
        await loadExistingData();

        // Set up event listeners
        setupEventListeners();

        // Check for saved model
        await checkSavedModel();

        // Update UI
        updateStats();
        updateTrainingButtonState();

        showToast('Application initialized successfully!', 'success');
    } catch (error) {
        console.error('Initialization error:', error);
        showToast('Failed to initialize application', 'error');
    }
}

function initCharts() {
    state.lossChart = createChart(
        'loss-chart',
        'line',
        'Loss',
        'rgb(220, 53, 69)',
        'rgba(220, 53, 69, 0.1)'
    );

    state.accuracyChart = createChart(
        'accuracy-chart',
        'line',
        'Accuracy',
        'rgb(40, 167, 69)',
        'rgba(40, 167, 69, 0.1)'
    );

    state.probabilityChart = createProbabilityChart('probability-chart');
}

async function loadExistingData() {
    const classes = await storage.getClasses();
    state.classes = classes;

    for (const cls of classes) {
        const images = await storage.getImagesByClass(cls.id);
        renderClassCard(cls, images);
    }

    updateClassSelector();
}

async function checkSavedModel() {
    const hasSavedModel = await modelManager.hasSavedModel();
    const modelInfo = await storage.getModelInfo();

    if (hasSavedModel && modelInfo) {
        elements.modelStatus.textContent = 'Saved Model Available';
        elements.loadModelBtn.disabled = false;
    }
}

// ==================== Event Listeners ====================
function setupEventListeners() {
    // Class management
    elements.addClassBtn.addEventListener('click', handleAddClass);
    elements.newClassName.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleAddClass();
    });

    // File drop zone
    setupDropZone(elements.dropZone, elements.fileInput, handleTrainingImages);

    // Test image drop zone
    setupDropZone(elements.testDropZone, elements.testFileInput, handleTestImage);

    // Parameter sliders
    elements.learningRate.addEventListener('input', () => {
        elements.lrValue.textContent = elements.learningRate.value;
    });
    elements.epochs.addEventListener('input', () => {
        elements.epochsValue.textContent = elements.epochs.value;
    });

    // Training controls
    elements.startTrainingBtn.addEventListener('click', handleStartTraining);
    elements.pauseTrainingBtn.addEventListener('click', handlePauseTraining);
    elements.stopTrainingBtn.addEventListener('click', handleStopTraining);

    // Model actions
    elements.saveModelBtn.addEventListener('click', handleSaveModel);
    elements.loadModelBtn.addEventListener('click', handleLoadModel);
    elements.clearDataBtn.addEventListener('click', handleClearData);

    // Target class selector
    elements.targetClass.addEventListener('change', (e) => {
        state.selectedClassId = parseInt(e.target.value);
    });
}

function setupDropZone(dropZone, fileInput, handler) {
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', async (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const files = Array.from(e.dataTransfer.files).filter(isImageFile);
        if (files.length > 0) {
            await handler(files);
        }
    });

    fileInput.addEventListener('change', async (e) => {
        const files = Array.from(e.target.files).filter(isImageFile);
        if (files.length > 0) {
            await handler(files);
        }
        fileInput.value = '';
    });
}

// ==================== Class Management ====================
async function handleAddClass() {
    const name = elements.newClassName.value.trim();
    if (!name) {
        showToast('Please enter a class name', 'warning');
        return;
    }

    // Check for duplicate names
    if (state.classes.some(c => c.name.toLowerCase() === name.toLowerCase())) {
        showToast('A class with this name already exists', 'warning');
        return;
    }

    try {
        const id = await storage.addClass(name);
        const newClass = { id, name };
        state.classes.push(newClass);

        renderClassCard(newClass, []);
        updateClassSelector();
        updateStats();
        updateTrainingButtonState();

        elements.newClassName.value = '';
        showToast(`Class "${name}" added successfully`, 'success');
    } catch (error) {
        console.error('Error adding class:', error);
        showToast('Failed to add class', 'error');
    }
}

function renderClassCard(cls, images) {
    const card = document.createElement('div');
    card.className = 'class-card';
    card.dataset.classId = cls.id;

    card.innerHTML = `
        <div class="class-header">
            <span class="class-name" data-id="${cls.id}">${cls.name}</span>
            <div class="class-actions">
                <button class="btn btn-small btn-secondary edit-class-btn" title="Edit name">✏️</button>
                <button class="btn btn-small btn-danger delete-class-btn" title="Delete class">🗑️</button>
            </div>
        </div>
        <div class="class-images" data-class-id="${cls.id}"></div>
        <div class="class-count"><span class="image-count">0</span> images</div>
    `;

    // Add event listeners
    const editBtn = card.querySelector('.edit-class-btn');
    const deleteBtn = card.querySelector('.delete-class-btn');
    const nameSpan = card.querySelector('.class-name');

    editBtn.addEventListener('click', () => handleEditClassName(cls.id, nameSpan));
    deleteBtn.addEventListener('click', () => handleDeleteClass(cls.id));

    // Make card selectable for adding images
    card.addEventListener('click', (e) => {
        if (!e.target.closest('.class-actions') && !e.target.closest('.class-image-container')) {
            selectClass(cls.id);
        }
    });

    elements.classesContainer.appendChild(card);

    // Render existing images
    const imagesContainer = card.querySelector('.class-images');
    images.forEach(img => renderImagePreview(imagesContainer, img, cls.id));
    updateImageCount(cls.id, images.length);
}

function handleEditClassName(classId, nameSpan) {
    const currentName = nameSpan.textContent;
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'class-name-input';
    input.value = currentName;

    nameSpan.replaceWith(input);
    input.focus();
    input.select();

    const saveEdit = async () => {
        const newName = input.value.trim();
        if (newName && newName !== currentName) {
            try {
                await storage.updateClassName(classId, newName);
                const cls = state.classes.find(c => c.id === classId);
                if (cls) cls.name = newName;
                updateClassSelector();
                showToast('Class name updated', 'success');
            } catch (error) {
                console.error('Error updating class name:', error);
                showToast('Failed to update class name', 'error');
            }
        }

        const span = document.createElement('span');
        span.className = 'class-name';
        span.dataset.id = classId;
        span.textContent = newName || currentName;
        input.replaceWith(span);
    };

    input.addEventListener('blur', saveEdit);
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            input.blur();
        }
    });
}

async function handleDeleteClass(classId) {
    if (!confirm('Are you sure you want to delete this class and all its images?')) {
        return;
    }

    try {
        await storage.deleteClass(classId);
        state.classes = state.classes.filter(c => c.id !== classId);

        const card = document.querySelector(`.class-card[data-class-id="${classId}"]`);
        if (card) card.remove();

        if (state.selectedClassId === classId) {
            state.selectedClassId = null;
        }

        updateClassSelector();
        updateStats();
        updateTrainingButtonState();

        showToast('Class deleted successfully', 'success');
    } catch (error) {
        console.error('Error deleting class:', error);
        showToast('Failed to delete class', 'error');
    }
}

function selectClass(classId) {
    // Remove selection from all cards
    document.querySelectorAll('.class-card').forEach(card => {
        card.classList.remove('selected');
    });

    // Select the clicked card
    const card = document.querySelector(`.class-card[data-class-id="${classId}"]`);
    if (card) {
        card.classList.add('selected');
    }

    state.selectedClassId = classId;
    elements.targetClass.value = classId;
    elements.classSelector.classList.remove('hidden');
}

function updateClassSelector() {
    elements.targetClass.innerHTML = '';
    state.classes.forEach(cls => {
        const option = document.createElement('option');
        option.value = cls.id;
        option.textContent = cls.name;
        elements.targetClass.appendChild(option);
    });

    if (state.classes.length > 0) {
        if (!state.selectedClassId || !state.classes.find(c => c.id === state.selectedClassId)) {
            state.selectedClassId = state.classes[0].id;
        }
        elements.targetClass.value = state.selectedClassId;
        elements.classSelector.classList.remove('hidden');
    } else {
        elements.classSelector.classList.add('hidden');
    }
}

// ==================== Image Management ====================
async function handleTrainingImages(files) {
    if (state.classes.length === 0) {
        showToast('Please add a class first', 'warning');
        return;
    }

    if (!state.selectedClassId) {
        state.selectedClassId = state.classes[0].id;
        selectClass(state.selectedClassId);
    }

    const classId = state.selectedClassId;
    const imagesContainer = document.querySelector(`.class-images[data-class-id="${classId}"]`);

    let addedCount = 0;
    for (const file of files) {
        try {
            const dataUrl = await readFileAsDataURL(file);
            const id = await storage.addImage(classId, dataUrl);
            renderImagePreview(imagesContainer, { id, data: dataUrl }, classId);
            addedCount++;
        } catch (error) {
            console.error('Error adding image:', error);
        }
    }

    // Update image count
    const currentImages = await storage.getImagesByClass(classId);
    updateImageCount(classId, currentImages.length);
    updateStats();
    updateTrainingButtonState();

    showToast(`Added ${addedCount} image(s) to training data`, 'success');
}

function renderImagePreview(container, imageData, classId) {
    const wrapper = document.createElement('div');
    wrapper.className = 'class-image-container';
    wrapper.dataset.imageId = imageData.id;

    const img = document.createElement('img');
    img.className = 'class-image-preview';
    img.src = imageData.data;
    img.alt = 'Training image';

    const deleteBtn = document.createElement('button');
    deleteBtn.className = 'delete-image';
    deleteBtn.textContent = '×';
    deleteBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        await handleDeleteImage(imageData.id, classId);
        wrapper.remove();
    });

    wrapper.appendChild(img);
    wrapper.appendChild(deleteBtn);
    container.appendChild(wrapper);
}

async function handleDeleteImage(imageId, classId) {
    try {
        await storage.deleteImage(imageId);
        const currentImages = await storage.getImagesByClass(classId);
        updateImageCount(classId, currentImages.length);
        updateStats();
        updateTrainingButtonState();
    } catch (error) {
        console.error('Error deleting image:', error);
        showToast('Failed to delete image', 'error');
    }
}

function updateImageCount(classId, count) {
    const card = document.querySelector(`.class-card[data-class-id="${classId}"]`);
    if (card) {
        const countSpan = card.querySelector('.image-count');
        if (countSpan) {
            countSpan.textContent = count;
        }
    }
}

// ==================== Training ====================
async function handleStartTraining() {
    if (state.isTraining) return;

    try {
        // Prepare training data
        const summary = await storage.getDataSummary();

        if (summary.numClasses < 2) {
            showToast('You need at least 2 classes to train', 'warning');
            return;
        }

        if (summary.totalImages < 4) {
            showToast('You need at least 4 images to train', 'warning');
            return;
        }

        // Get parameters
        const params = {
            learningRate: parseFloat(elements.learningRate.value),
            batchSize: parseInt(elements.batchSize.value),
            epochs: parseInt(elements.epochs.value),
            architecture: elements.modelArchitecture.value
        };

        // Prepare data
        const classesWithImages = {};
        for (const cls of state.classes) {
            const images = await storage.getImagesByClass(cls.id);
            if (images.length > 0) {
                classesWithImages[cls.id] = {
                    name: cls.name,
                    images
                };
            }
        }

        showToast('Preparing training data...', 'info');
        const trainingData = await modelManager.prepareTrainingData(classesWithImages);

        // Reset charts
        resetChart(state.lossChart);
        resetChart(state.accuracyChart);

        // Update UI
        state.isTraining = true;
        state.isPaused = false;
        state.trainingStartTime = Date.now();
        updateTrainingUI('training');

        // Start timer
        startTimer();

        // Start training
        await modelManager.train(trainingData, params, {
            onEpochEnd: (epoch, totalEpochs, logs) => {
                updateChart(state.lossChart, epoch + 1, logs.loss);
                updateChart(state.accuracyChart, epoch + 1, logs.acc);

                elements.currentEpoch.textContent = `${epoch + 1} / ${totalEpochs}`;
                elements.trainingProgress.style.width = `${((epoch + 1) / totalEpochs) * 100}%`;

                // Estimate remaining time
                const elapsed = (Date.now() - state.trainingStartTime) / 1000;
                const perEpoch = elapsed / (epoch + 1);
                const remaining = perEpoch * (totalEpochs - epoch - 1);
                elements.remainingTime.textContent = formatTime(remaining);
            },
            onBatchEnd: (batch, totalBatches, logs) => {
                elements.batchProgress.textContent = `${batch + 1} / ${totalBatches}`;
            },
            onTrainingEnd: (completed) => {
                state.isTraining = false;
                stopTimer();
                trainingData.xs.dispose();
                trainingData.ys.dispose();

                if (completed) {
                    elements.modelStatus.textContent = 'Trained';
                    elements.saveModelBtn.disabled = false;
                    showToast('Training completed successfully!', 'success');

                    // Update final accuracy
                    const lastAcc = state.accuracyChart.data.datasets[0].data;
                    if (lastAcc.length > 0) {
                        elements.finalAccuracy.textContent = formatPercent(lastAcc[lastAcc.length - 1]);
                    }
                } else {
                    showToast('Training stopped', 'warning');
                }

                updateTrainingUI('idle');
            },
            onTrainingError: (error) => {
                console.error('Training error:', error);
                state.isTraining = false;
                stopTimer();
                trainingData.xs.dispose();
                trainingData.ys.dispose();
                showToast('Training failed: ' + error.message, 'error');
                updateTrainingUI('idle');
            }
        });
    } catch (error) {
        console.error('Error starting training:', error);
        state.isTraining = false;
        showToast('Failed to start training: ' + error.message, 'error');
        updateTrainingUI('idle');
    }
}

function handlePauseTraining() {
    if (!state.isTraining) return;

    if (state.isPaused) {
        modelManager.resumeTraining();
        state.isPaused = false;
        elements.pauseTrainingBtn.textContent = '⏸️ Pause';
        elements.trainingStatus.textContent = 'Training...';
    } else {
        modelManager.pauseTraining();
        state.isPaused = true;
        elements.pauseTrainingBtn.textContent = '▶️ Resume';
        elements.trainingStatus.textContent = 'Paused';
    }
}

function handleStopTraining() {
    if (!state.isTraining) return;

    if (confirm('Are you sure you want to stop training?')) {
        modelManager.stopTraining();
    }
}

function updateTrainingUI(status) {
    switch (status) {
        case 'training':
            elements.trainingStatus.textContent = 'Training...';
            elements.startTrainingBtn.disabled = true;
            elements.pauseTrainingBtn.disabled = false;
            elements.stopTrainingBtn.disabled = false;
            elements.trainingStatus.classList.add('training-active');
            disableParamInputs(true);
            break;
        case 'idle':
            elements.trainingStatus.textContent = 'Idle';
            elements.startTrainingBtn.disabled = false;
            elements.pauseTrainingBtn.disabled = true;
            elements.stopTrainingBtn.disabled = true;
            elements.pauseTrainingBtn.textContent = '⏸️ Pause';
            elements.trainingStatus.classList.remove('training-active');
            disableParamInputs(false);
            updateTrainingButtonState();
            break;
    }
}

function disableParamInputs(disabled) {
    elements.learningRate.disabled = disabled;
    elements.batchSize.disabled = disabled;
    elements.epochs.disabled = disabled;
    elements.modelArchitecture.disabled = disabled;
}

function startTimer() {
    if (state.timerInterval) {
        clearInterval(state.timerInterval);
    }

    state.timerInterval = setInterval(() => {
        const elapsed = (Date.now() - state.trainingStartTime) / 1000;
        elements.elapsedTime.textContent = formatTime(elapsed);
    }, 1000);
}

function stopTimer() {
    if (state.timerInterval) {
        clearInterval(state.timerInterval);
        state.timerInterval = null;
    }
}

// ==================== Validation ====================
async function handleTestImage(files) {
    if (files.length === 0) return;

    if (!modelManager.model) {
        showToast('Please train or load a model first', 'warning');
        return;
    }

    try {
        const file = files[0];
        const dataUrl = await readFileAsDataURL(file);

        // Display preview
        elements.testImagePreview.src = dataUrl;
        elements.predictionResults.classList.remove('hidden');

        // Create image element for prediction
        const img = new Image();
        img.onload = async () => {
            const predictions = await modelManager.predict(img);

            // Update predicted class
            const topPrediction = predictions[0];
            elements.predictedClass.textContent =
                `${topPrediction.className}: ${formatPercent(topPrediction.probability)}`;

            // Update probability chart
            updateProbabilityChart(state.probabilityChart, predictions);
        };
        img.src = dataUrl;
    } catch (error) {
        console.error('Error testing image:', error);
        showToast('Failed to test image', 'error');
    }
}

// ==================== Model Management ====================
async function handleSaveModel() {
    if (!modelManager.model) {
        showToast('No model to save', 'warning');
        return;
    }

    try {
        const modelInfo = await modelManager.saveModel();
        await storage.saveModelInfo(modelInfo);
        showToast('Model saved successfully!', 'success');
    } catch (error) {
        console.error('Error saving model:', error);
        showToast('Failed to save model', 'error');
    }
}

async function handleLoadModel() {
    try {
        const modelInfo = await storage.getModelInfo();
        if (!modelInfo) {
            showToast('No saved model found', 'warning');
            return;
        }

        await modelManager.loadModel(modelInfo);
        elements.modelStatus.textContent = 'Loaded';
        elements.saveModelBtn.disabled = false;
        showToast('Model loaded successfully!', 'success');
    } catch (error) {
        console.error('Error loading model:', error);
        showToast('Failed to load model', 'error');
    }
}

async function handleClearData() {
    if (!confirm('Are you sure you want to clear all data? This will delete all classes, images, and saved models.')) {
        return;
    }

    try {
        await storage.clearAll();
        await modelManager.deleteModel();

        // Clear UI
        elements.classesContainer.innerHTML = '';
        state.classes = [];
        state.selectedClassId = null;

        // Reset charts
        resetChart(state.lossChart);
        resetChart(state.accuracyChart);

        // Reset stats
        updateStats();
        updateTrainingButtonState();

        elements.modelStatus.textContent = 'Not Trained';
        elements.saveModelBtn.disabled = true;
        elements.finalAccuracy.textContent = '--';
        elements.predictionResults.classList.add('hidden');
        elements.classSelector.classList.add('hidden');

        showToast('All data cleared', 'success');
    } catch (error) {
        console.error('Error clearing data:', error);
        showToast('Failed to clear data', 'error');
    }
}

// ==================== Stats ====================
async function updateStats() {
    try {
        const summary = await storage.getDataSummary();
        elements.totalImages.textContent = summary.totalImages;
        elements.numClasses.textContent = summary.numClasses;
    } catch (error) {
        console.error('Error updating stats:', error);
    }
}

function updateTrainingButtonState() {
    const canTrain = state.classes.length >= 2;
    elements.startTrainingBtn.disabled = !canTrain || state.isTraining;
}

// ==================== Initialize Application ====================
document.addEventListener('DOMContentLoaded', init);
