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

## Why ResNet50?

ResNet50 was chosen as the backbone architecture for several key reasons:

1. **Proven Performance**: ResNet50 has demonstrated excellent performance on image classification tasks and is widely used in production systems. It won the ImageNet challenge in 2015 and has since become a standard benchmark in computer vision.

2. **Residual Connections**: The architecture's residual connections help combat the vanishing gradient problem, allowing for better training of deeper networks. This is particularly important for learning complex features in food images.

3. **Transfer Learning Benefits**: 
   - Pre-trained on ImageNet, which contains many food-related images
   - Rich feature extraction capabilities that can be fine-tuned for food freshness detection
   - Good balance between model complexity and performance

4. **Computational Efficiency**:
   - More efficient than larger models like ResNet101 or ResNet152
   - Better suited for CPU-based training and inference
   - Faster training times while maintaining good accuracy

5. **Memory Requirements**:
   - Moderate memory footprint compared to larger architectures
   - Suitable for systems with limited computational resources
   - Efficient for deployment in production environments

Alternative architectures considered:
- **VGG**: Too many parameters, making it slower to train and more memory-intensive
- **MobileNet**: Less accurate for complex food classification tasks
- **EfficientNet**: More complex architecture, potentially overkill for binary classification
- **DenseNet**: Higher memory requirements and slower inference times

## Activation Functions: ReLU and Sigmoid

The model uses two different activation functions for specific purposes:

### ReLU (Rectified Linear Unit)
- Used in the hidden layer (after the first linear layer)
- Advantages:
  - **Computational Efficiency**: Simple max(0,x) operation, making it faster to compute
  - **Sparsity**: Can create sparse representations, helping with feature selection
  - **Reduced Vanishing Gradient**: Less prone to the vanishing gradient problem compared to sigmoid/tanh
  - **Biological Plausibility**: More closely mimics the behavior of biological neurons

### Sigmoid
- Used in the final layer for binary classification
- Advantages:
  - **Output Range**: Produces outputs between 0 and 1, perfect for binary classification
  - **Probabilistic Interpretation**: Output can be directly interpreted as probability
  - **Smooth Gradient**: Provides smooth gradients for backpropagation
  - **Decision Boundary**: Creates a clear decision boundary at 0.5

Why not other activation functions?
- **Tanh**: While similar to sigmoid, its output range (-1 to 1) is less suitable for binary classification
- **Softmax**: More appropriate for multi-class problems, overkill for binary classification
- **Leaky ReLU**: Could be used instead of ReLU, but standard ReLU works well for this task
- **ELU/SELU**: More complex, with no significant advantage for this specific use case

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