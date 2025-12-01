# TensorFlow.js Image Classifier

A pure frontend image classification web application using TensorFlow.js that runs entirely in the browser without a backend.

## Features

### Training Data Management
- Drag-and-drop/click area for uploading training images
- Dynamic class management: Add, delete, and rename classes
- Image preview and classification labeling functionality

### Training Parameter Configuration
- Learning Rate selector (slider, range 0.0001-0.01)
- Batch Size settings (options: 16, 32, 64)
- Epochs settings (range 10-100)
- Model Architecture selection (Simple CNN, MobileNet Transfer Learning)

### Training Control Panel
- Start/Pause/Stop training buttons
- Real-time training progress display
- Current epoch and batch progress indicators
- Elapsed and remaining time display

### Training Visualization
- Real-time Loss and Accuracy curves using Chart.js
- Training status indicators
- Progress bar

### Model Validation Interface
- Test image upload area
- Prediction results with:
  - Preview of uploaded image
  - Probability bar chart for each class
  - Highlighting the class with the highest probability
- Model performance statistics

## Technical Details

- **Pure Frontend**: No backend required - runs entirely in the browser
- **TensorFlow.js**: Used for training and inference
- **IndexedDB**: Stores training data and model weights persistently
- **Responsive Design**: Works on Desktop and Mobile devices
- **Chart.js**: Visualizes training progress

## Getting Started

1. Open `index.html` in a modern web browser
2. Add at least 2 classes using the "Add Class" button
3. Select a class and drag-and-drop or click to upload training images
4. Configure training parameters (learning rate, batch size, epochs, architecture)
5. Click "Start Training" to begin training
6. Once trained, upload a test image to see predictions

## Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge

## Files Structure

```
├── index.html          # Main HTML page
├── css/
│   └── styles.css      # Styling and responsive design
├── js/
│   ├── app.js          # Main application logic
│   ├── model.js        # TensorFlow.js model management
│   ├── storage.js      # IndexedDB storage utilities
│   └── ui.js           # UI utilities
└── README.md           # This file
```

## License

MIT