/**
 * TensorFlow.js Model Module
 * Handles model creation, training, and inference
 */

// Image dimensions for training
const IMAGE_SIZE = 224;

class ModelManager {
    constructor() {
        this.model = null;
        this.mobileNetBase = null;
        this.classLabels = [];
        this.isTraining = false;
        this.isPaused = false;
        this.shouldStop = false;
        this.currentEpoch = 0;
        this.trainingCallbacks = {};
    }

    /**
     * Create a simple CNN model
     */
    createSimpleCNN(numClasses) {
        const model = tf.sequential();

        // First convolutional block
        model.add(tf.layers.conv2d({
            inputShape: [IMAGE_SIZE, IMAGE_SIZE, 3],
            filters: 32,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same'
        }));
        model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

        // Second convolutional block
        model.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same'
        }));
        model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

        // Third convolutional block
        model.add(tf.layers.conv2d({
            filters: 128,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same'
        }));
        model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

        // Flatten and dense layers
        model.add(tf.layers.flatten());
        model.add(tf.layers.dropout({ rate: 0.5 }));
        model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
        model.add(tf.layers.dropout({ rate: 0.3 }));
        model.add(tf.layers.dense({ units: numClasses, activation: 'softmax' }));

        return model;
    }

    /**
     * Load MobileNet base model for transfer learning
     */
    async loadMobileNetBase() {
        if (this.mobileNetBase) {
            return this.mobileNetBase;
        }

        // Load MobileNet
        const mobilenet = await tf.loadLayersModel(
            'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'
        );

        // Get the layer before the final classification layer
        const layer = mobilenet.getLayer('conv_pw_13_relu');
        this.mobileNetBase = tf.model({
            inputs: mobilenet.inputs,
            outputs: layer.output
        });

        // Freeze base model weights
        for (const layer of this.mobileNetBase.layers) {
            layer.trainable = false;
        }

        return this.mobileNetBase;
    }

    /**
     * Create a MobileNet transfer learning model
     */
    async createMobileNetModel(numClasses) {
        await this.loadMobileNetBase();

        // Create the classification head
        const input = tf.input({ shape: [IMAGE_SIZE, IMAGE_SIZE, 3] });
        const baseOutput = this.mobileNetBase.apply(input);

        let x = tf.layers.globalAveragePooling2d().apply(baseOutput);
        x = tf.layers.dropout({ rate: 0.5 }).apply(x);
        x = tf.layers.dense({ units: 128, activation: 'relu' }).apply(x);
        x = tf.layers.dropout({ rate: 0.3 }).apply(x);
        const output = tf.layers.dense({ units: numClasses, activation: 'softmax' }).apply(x);

        return tf.model({ inputs: input, outputs: output });
    }

    /**
     * Create model based on selected architecture
     */
    async createModel(architecture, numClasses) {
        this.classLabels = [];

        if (architecture === 'mobilenet') {
            this.model = await this.createMobileNetModel(numClasses);
        } else {
            this.model = this.createSimpleCNN(numClasses);
        }

        return this.model;
    }

    /**
     * Compile the model with specified learning rate
     */
    compileModel(learningRate) {
        if (!this.model) {
            throw new Error('Model not created');
        }

        this.model.compile({
            optimizer: tf.train.adam(learningRate),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
    }

    /**
     * Preprocess an image for training/inference
     */
    preprocessImage(imageElement) {
        return tf.tidy(() => {
            // Convert image to tensor
            let tensor = tf.browser.fromPixels(imageElement);

            // Resize to model input size
            tensor = tf.image.resizeBilinear(tensor, [IMAGE_SIZE, IMAGE_SIZE]);

            // Normalize to [0, 1]
            tensor = tensor.div(255.0);

            // Add batch dimension
            return tensor.expandDims(0);
        });
    }

    /**
     * Prepare training data from image data
     */
    async prepareTrainingData(classesWithImages) {
        const images = [];
        const labels = [];
        this.classLabels = [];

        let classIndex = 0;
        for (const [classId, classData] of Object.entries(classesWithImages)) {
            this.classLabels.push({ id: parseInt(classId), name: classData.name, index: classIndex });

            for (const imageData of classData.images) {
                // Create an image element from the data URL
                const img = await this.loadImage(imageData.data);
                images.push(img);
                labels.push(classIndex);
            }
            classIndex++;
        }

        // Convert to tensors
        const numClasses = this.classLabels.length;
        const xs = tf.tidy(() => {
            const tensors = images.map(img => this.preprocessImage(img).squeeze());
            return tf.stack(tensors);
        });

        const ys = tf.tidy(() => {
            return tf.oneHot(tf.tensor1d(labels, 'int32'), numClasses);
        });

        return { xs, ys, numClasses };
    }

    /**
     * Load an image from a data URL
     */
    loadImage(dataUrl) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = dataUrl;
        });
    }

    /**
     * Train the model
     */
    async train(trainingData, params, callbacks) {
        const { xs, ys, numClasses } = trainingData;
        const { learningRate, batchSize, epochs, architecture } = params;

        this.isTraining = true;
        this.isPaused = false;
        this.shouldStop = false;
        this.currentEpoch = 0;
        this.trainingCallbacks = callbacks;

        try {
            // Create and compile model
            await this.createModel(architecture, numClasses);
            this.compileModel(learningRate);

            // Calculate total batches
            const numSamples = xs.shape[0];
            const totalBatches = Math.ceil(numSamples / batchSize);

            // Training loop with pause/stop support
            for (let epoch = 0; epoch < epochs && !this.shouldStop; epoch++) {
                this.currentEpoch = epoch;

                // Wait while paused
                while (this.isPaused && !this.shouldStop) {
                    await new Promise(resolve => setTimeout(resolve, 100));
                }

                if (this.shouldStop) break;

                // Train one epoch
                const history = await this.model.fit(xs, ys, {
                    batchSize,
                    epochs: 1,
                    shuffle: true,
                    validationSplit: 0.2,
                    callbacks: {
                        onBatchEnd: async (batch, logs) => {
                            if (callbacks.onBatchEnd) {
                                callbacks.onBatchEnd(batch, totalBatches, logs);
                            }
                            // Allow UI updates
                            await tf.nextFrame();
                        }
                    }
                });

                // Report epoch end
                if (callbacks.onEpochEnd) {
                    callbacks.onEpochEnd(epoch, epochs, {
                        loss: history.history.loss[0],
                        acc: history.history.acc[0],
                        val_loss: history.history.val_loss?.[0],
                        val_acc: history.history.val_acc?.[0]
                    });
                }
            }

            this.isTraining = false;
            if (callbacks.onTrainingEnd) {
                callbacks.onTrainingEnd(!this.shouldStop);
            }

            return !this.shouldStop;
        } catch (error) {
            this.isTraining = false;
            if (callbacks.onTrainingError) {
                callbacks.onTrainingError(error);
            }
            throw error;
        }
    }

    /**
     * Pause training
     */
    pauseTraining() {
        this.isPaused = true;
    }

    /**
     * Resume training
     */
    resumeTraining() {
        this.isPaused = false;
    }

    /**
     * Stop training
     */
    stopTraining() {
        this.shouldStop = true;
        this.isPaused = false;
    }

    /**
     * Make a prediction on an image
     */
    async predict(imageElement) {
        if (!this.model) {
            throw new Error('Model not trained');
        }

        return tf.tidy(() => {
            const tensor = this.preprocessImage(imageElement);
            const predictions = this.model.predict(tensor);
            const probabilities = predictions.dataSync();

            return this.classLabels.map((cls, i) => ({
                classId: cls.id,
                className: cls.name,
                probability: probabilities[i]
            })).sort((a, b) => b.probability - a.probability);
        });
    }

    /**
     * Save the model to IndexedDB
     */
    async saveModel() {
        if (!this.model) {
            throw new Error('No model to save');
        }

        await this.model.save('indexeddb://image-classifier-model');

        return {
            classLabels: this.classLabels,
            savedAt: Date.now()
        };
    }

    /**
     * Load a model from IndexedDB
     */
    async loadModel(modelInfo) {
        this.model = await tf.loadLayersModel('indexeddb://image-classifier-model');
        this.classLabels = modelInfo.classLabels || [];

        return this.model;
    }

    /**
     * Check if a saved model exists
     */
    async hasSavedModel() {
        try {
            const models = await tf.io.listModels();
            return 'indexeddb://image-classifier-model' in models;
        } catch {
            return false;
        }
    }

    /**
     * Delete saved model
     */
    async deleteModel() {
        try {
            await tf.io.removeModel('indexeddb://image-classifier-model');
        } catch {
            // Model might not exist
        }
        this.model = null;
        this.classLabels = [];
    }

    /**
     * Get model summary
     */
    getModelSummary() {
        if (!this.model) {
            return null;
        }

        const layers = [];
        this.model.layers.forEach(layer => {
            layers.push({
                name: layer.name,
                type: layer.getClassName(),
                outputShape: layer.outputShape
            });
        });

        return {
            totalParams: this.model.countParams(),
            layers
        };
    }

    /**
     * Dispose of tensors and model
     */
    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
        if (this.mobileNetBase) {
            this.mobileNetBase.dispose();
            this.mobileNetBase = null;
        }
    }
}

// Export singleton instance
export const modelManager = new ModelManager();
export { IMAGE_SIZE };
