/**
 * IndexedDB Storage Module
 * Handles persistent storage for training data and model weights
 */

const DB_NAME = 'TensorFlowImageClassifier';
const DB_VERSION = 1;
const STORES = {
    CLASSES: 'classes',
    IMAGES: 'images',
    MODELS: 'models',
    SETTINGS: 'settings'
};

class StorageManager {
    constructor() {
        this.db = null;
    }

    /**
     * Initialize IndexedDB
     */
    async init() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(DB_NAME, DB_VERSION);

            request.onerror = () => {
                reject(new Error('Failed to open IndexedDB'));
            };

            request.onsuccess = (event) => {
                this.db = event.target.result;
                resolve(this);
            };

            request.onupgradeneeded = (event) => {
                const db = event.target.result;

                // Create object stores
                if (!db.objectStoreNames.contains(STORES.CLASSES)) {
                    db.createObjectStore(STORES.CLASSES, { keyPath: 'id', autoIncrement: true });
                }

                if (!db.objectStoreNames.contains(STORES.IMAGES)) {
                    const imageStore = db.createObjectStore(STORES.IMAGES, { keyPath: 'id', autoIncrement: true });
                    imageStore.createIndex('classId', 'classId', { unique: false });
                }

                if (!db.objectStoreNames.contains(STORES.MODELS)) {
                    db.createObjectStore(STORES.MODELS, { keyPath: 'id' });
                }

                if (!db.objectStoreNames.contains(STORES.SETTINGS)) {
                    db.createObjectStore(STORES.SETTINGS, { keyPath: 'key' });
                }
            };
        });
    }

    /**
     * Get a transaction for specified store
     */
    getTransaction(storeName, mode = 'readonly') {
        return this.db.transaction(storeName, mode);
    }

    /**
     * Get an object store
     */
    getStore(storeName, mode = 'readonly') {
        return this.getTransaction(storeName, mode).objectStore(storeName);
    }

    // ==================== Class Operations ====================

    /**
     * Add a new class
     */
    async addClass(name) {
        return new Promise((resolve, reject) => {
            const store = this.getStore(STORES.CLASSES, 'readwrite');
            const request = store.add({ name, createdAt: Date.now() });

            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    /**
     * Get all classes
     */
    async getClasses() {
        return new Promise((resolve, reject) => {
            const store = this.getStore(STORES.CLASSES);
            const request = store.getAll();

            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    /**
     * Update a class name
     */
    async updateClassName(id, name) {
        return new Promise((resolve, reject) => {
            const store = this.getStore(STORES.CLASSES, 'readwrite');
            const getRequest = store.get(id);

            getRequest.onsuccess = () => {
                const data = getRequest.result;
                if (data) {
                    data.name = name;
                    const updateRequest = store.put(data);
                    updateRequest.onsuccess = () => resolve(true);
                    updateRequest.onerror = () => reject(updateRequest.error);
                } else {
                    reject(new Error('Class not found'));
                }
            };
            getRequest.onerror = () => reject(getRequest.error);
        });
    }

    /**
     * Delete a class and its images
     */
    async deleteClass(id) {
        // First delete all images for this class
        await this.deleteImagesByClass(id);

        return new Promise((resolve, reject) => {
            const store = this.getStore(STORES.CLASSES, 'readwrite');
            const request = store.delete(id);

            request.onsuccess = () => resolve(true);
            request.onerror = () => reject(request.error);
        });
    }

    // ==================== Image Operations ====================

    /**
     * Add an image to a class
     */
    async addImage(classId, imageData) {
        return new Promise((resolve, reject) => {
            const store = this.getStore(STORES.IMAGES, 'readwrite');
            const request = store.add({
                classId,
                data: imageData,
                createdAt: Date.now()
            });

            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    /**
     * Get all images for a class
     */
    async getImagesByClass(classId) {
        return new Promise((resolve, reject) => {
            const store = this.getStore(STORES.IMAGES);
            const index = store.index('classId');
            const request = index.getAll(classId);

            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    /**
     * Get all images
     */
    async getAllImages() {
        return new Promise((resolve, reject) => {
            const store = this.getStore(STORES.IMAGES);
            const request = store.getAll();

            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    /**
     * Delete an image
     */
    async deleteImage(id) {
        return new Promise((resolve, reject) => {
            const store = this.getStore(STORES.IMAGES, 'readwrite');
            const request = store.delete(id);

            request.onsuccess = () => resolve(true);
            request.onerror = () => reject(request.error);
        });
    }

    /**
     * Delete all images for a class
     */
    async deleteImagesByClass(classId) {
        const images = await this.getImagesByClass(classId);
        const store = this.getStore(STORES.IMAGES, 'readwrite');

        return Promise.all(images.map(img => {
            return new Promise((resolve, reject) => {
                const request = store.delete(img.id);
                request.onsuccess = () => resolve(true);
                request.onerror = () => reject(request.error);
            });
        }));
    }

    // ==================== Model Operations ====================

    /**
     * Save model info
     */
    async saveModelInfo(modelInfo) {
        return new Promise((resolve, reject) => {
            const store = this.getStore(STORES.MODELS, 'readwrite');
            const request = store.put({
                id: 'current',
                ...modelInfo,
                savedAt: Date.now()
            });

            request.onsuccess = () => resolve(true);
            request.onerror = () => reject(request.error);
        });
    }

    /**
     * Get model info
     */
    async getModelInfo() {
        return new Promise((resolve, reject) => {
            const store = this.getStore(STORES.MODELS);
            const request = store.get('current');

            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    // ==================== Settings Operations ====================

    /**
     * Save a setting
     */
    async saveSetting(key, value) {
        return new Promise((resolve, reject) => {
            const store = this.getStore(STORES.SETTINGS, 'readwrite');
            const request = store.put({ key, value });

            request.onsuccess = () => resolve(true);
            request.onerror = () => reject(request.error);
        });
    }

    /**
     * Get a setting
     */
    async getSetting(key) {
        return new Promise((resolve, reject) => {
            const store = this.getStore(STORES.SETTINGS);
            const request = store.get(key);

            request.onsuccess = () => resolve(request.result?.value);
            request.onerror = () => reject(request.error);
        });
    }

    // ==================== Utility Operations ====================

    /**
     * Clear all data
     */
    async clearAll() {
        const storeNames = [STORES.CLASSES, STORES.IMAGES, STORES.MODELS, STORES.SETTINGS];

        for (const storeName of storeNames) {
            await new Promise((resolve, reject) => {
                const store = this.getStore(storeName, 'readwrite');
                const request = store.clear();

                request.onsuccess = () => resolve(true);
                request.onerror = () => reject(request.error);
            });
        }

        return true;
    }

    /**
     * Get training data summary
     */
    async getDataSummary() {
        const classes = await this.getClasses();
        const images = await this.getAllImages();

        const classCounts = {};
        for (const cls of classes) {
            classCounts[cls.id] = 0;
        }
        for (const img of images) {
            if (classCounts[img.classId] !== undefined) {
                classCounts[img.classId]++;
            }
        }

        return {
            numClasses: classes.length,
            totalImages: images.length,
            classCounts,
            classes
        };
    }
}

// Export singleton instance
export const storage = new StorageManager();
