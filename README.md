# Food Freshness Detection

A deep learning-based system for detecting whether food items are fresh or rotten using computer vision. This project uses a ResNet50-based model to classify food images into two categories: fresh and rotten.

## Features

- Binary classification of food images (fresh vs. rotten)
- Transfer learning using pretrained ResNet50 model
- Comprehensive training metrics visualization
- Support for both training and prediction modes
- Early stopping and learning rate scheduling
- CPU-optimized implementation

## Requirements

- Python 3.x
- PyTorch
- torchvision
- PIL (Python Imaging Library)
- NumPy
- scikit-learn
- matplotlib
- tqdm

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd food-freshness-detection
```

2. Install the required packages:
```bash
pip install torch torchvision pillow numpy scikit-learn matplotlib tqdm
```

## Dataset Structure

Organize your dataset in the following structure:
```
.
├── fresh/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── rotten/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Usage

### Training the Model

To train the model:
```bash
python food_freshness_detector.py --mode train
```

The training process will:
- Split the dataset into training (80%) and testing (20%) sets
- Train the model for up to 30 epochs with early stopping
- Save the best model as 'model.pth'
- Generate training metrics visualization as 'training_metrics.png'

### Making Predictions

To predict the freshness of a food image:
```bash
python food_freshness_detector.py --mode predict --image_path path/to/your/image.jpg
```

## Model Architecture

The model uses a ResNet50 backbone with the following modifications:
- Freezes all layers except the last 20
- Adds custom classification head with:
  - Linear layer (2048 → 512)
  - ReLU activation
  - Dropout (0.5)
  - Linear layer (512 → 1)
  - Sigmoid activation

## Training Details

- Optimizer: Adam with learning rate 0.001
- Loss Function: Binary Cross Entropy
- Learning Rate Scheduling: ReduceLROnPlateau
- Early Stopping: Patience of 10 epochs
- Batch Size: 16
- Image Size: 224x224
- Data Augmentation:
  - Random horizontal flips
  - Random rotations (±10 degrees)
  - Normalization using ImageNet statistics

## Performance Metrics

The model tracks and visualizes:
- Training and test loss
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Output Files

- `model.pth`: Saved model weights
- `training_metrics.png`: Visualization of training metrics

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Contact

[Add your contact information here] 