# Food Freshness Detection Using Deep Learning
## Project Presentation

---

## Outline
1. Introduction
2. Problem Statement
3. Technical Background
4. Methodology
5. Implementation
6. Results & Analysis
7. Future Work
8. Conclusion

---

## 1. Introduction
- Food waste is a global issue
- Manual inspection is time-consuming and subjective
- Need for automated, accurate freshness detection
- Potential applications in:
  - Food industry
  - Smart refrigerators
  - Quality control systems

---

## 2. Problem Statement
- Binary Classification Problem:
  - Input: Food images
  - Output: Fresh or Rotten
- Dataset:
  - Fresh images: 1,141 samples
  - Rotten images: 775 samples
- Challenge: Develop an accurate, efficient classification system

---

## 3. Technical Background

### Deep Learning & Computer Vision
- Convolutional Neural Networks (CNNs)
  - Specialized for image processing
  - Hierarchical feature extraction
  - Spatial pattern recognition
  - Key Components:
    ```python
    # CNN Building Blocks
    nn.Conv2d(in_channels, out_channels, kernel_size)  # Convolutional layer
    nn.MaxPool2d(kernel_size)                         # Pooling layer
    nn.ReLU()                                         # Activation function
    nn.BatchNorm2d(num_features)                      # Batch normalization
    ```

### Transfer Learning
- Why Transfer Learning?
  - Limited dataset size
  - Pre-trained models have learned useful features
  - Faster training and better performance
- ResNet50 Architecture
  - 50 layers deep
  - Residual connections
  - Pre-trained on ImageNet
  - Key Features:
    ```python
    # Residual Block Structure
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
            # Skip connection
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride),
                    nn.BatchNorm2d(out_channels)
                )
    ```

### Key Technologies
1. PyTorch
   - Dynamic computation graphs
   - GPU acceleration support
   - Rich ecosystem
   - Key Features:
     ```python
     # PyTorch DataLoader
     train_loader = DataLoader(
         dataset,
         batch_size=16,
         shuffle=True,
         num_workers=0
     )
     ```

2. Data Augmentation
   - Random horizontal flips
   - Random rotations
   - Normalization
   - Purpose: Prevent overfitting, increase robustness
   - Implementation:
     ```python
     transform = transforms.Compose([
         transforms.Resize((224, 224)),
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(10),
         transforms.ToTensor(),
         transforms.Normalize(
             mean=[0.485, 0.456, 0.406],
             std=[0.229, 0.224, 0.225]
         )
     ])
     ```

---

## 4. Methodology

### Model Architecture
- Base: ResNet50 (pre-trained)
- Modifications:
  - Frozen early layers
  - Custom classification head
  - Binary output (Fresh/Rotten)
- Implementation:
  ```python
  class FoodFreshnessModel(nn.Module):
      def __init__(self):
          super().__init__()
          # Load pretrained ResNet50
          self.model = models.resnet50(pretrained=True)
          
          # Freeze early layers
          for param in list(self.model.parameters())[:-20]:
              param.requires_grad = False
          
          # Custom classification head
          num_features = self.model.fc.in_features
          self.model.fc = nn.Sequential(
              nn.Linear(num_features, 512),
              nn.ReLU(),
              nn.Dropout(0.5),
              nn.Linear(512, 1),
              nn.Sigmoid()
          )
  ```

### Training Strategy
1. Data Preparation
   - 80% training, 20% testing split
   - Image resizing to 224x224
   - Data augmentation
   - Implementation:
     ```python
     # Dataset split
     train_size = int(0.8 * len(dataset))
     test_size = len(dataset) - train_size
     train_dataset, test_dataset = torch.utils.data.random_split(
         dataset, [train_size, test_size]
     )
     ```

2. Training Process
   - Loss Function: Binary Cross Entropy
   - Optimizer: Adam
   - Learning Rate: 0.001 with reduction on plateau
   - Early stopping (patience = 10)
   - Implementation:
     ```python
     # Training setup
     criterion = nn.BCELoss()
     optimizer = optim.Adam(model.parameters(), lr=0.001)
     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
         optimizer, mode='min', factor=0.1, patience=3
     )
     ```

3. Evaluation Metrics
   - Accuracy: Measures the overall correctness of predictions (correct predictions / total predictions)
   - Precision: Measures how many of the predicted positive cases were actually positive (true positives / (true positives + false positives))
   - Recall: Measures how many of the actual positive cases were correctly identified (true positives / (true positives + false negatives))
   - F1 Score: Harmonic mean of precision and recall, providing a balance between the two metrics
   - Confusion Matrix: A table showing the distribution of predictions across actual classes, helping identify where the model makes mistakes
   - Implementation:
     ```python
     def calculate_metrics(y_true, y_pred):
         accuracy = accuracy_score(y_true, y_pred)
         precision = precision_score(y_true, y_pred)
         recall = recall_score(y_true, y_pred)
         f1 = f1_score(y_true, y_pred)
         cm = confusion_matrix(y_true, y_pred)
         return accuracy, precision, recall, f1, cm
     ```

---

## 5. Implementation

### Code Structure
```python
class FoodDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['fresh', 'rotten']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        # Load images and labels
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls])
```

### Training Pipeline
1. Data Loading
2. Model Initialization
3. Training Loop
4. Evaluation
5. Model Saving
- Implementation:
  ```python
  def train_epoch(model, train_loader, criterion, optimizer, device):
      model.train()
      running_loss = 0.0
      for inputs, labels in train_loader:
          inputs, labels = inputs.to(device), labels.float().to(device)
          
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs.squeeze(), labels)
          loss.backward()
          optimizer.step()
          
          running_loss += loss.item()
      
      return running_loss / len(train_loader)
  ```

---

## 6. Results & Analysis

### Training Metrics
- Loss curves
- Accuracy progression
- Precision-Recall trade-off
- F1 Score trends
- Implementation:
  ```python
  def plot_metrics(metrics_history):
      epochs = range(1, len(metrics_history['train_loss']) + 1)
      fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
      
      # Plot losses
      ax1.plot(epochs, metrics_history['train_loss'], label='Train Loss')
      ax1.plot(epochs, metrics_history['test_loss'], label='Test Loss')
      ax1.set_title('Training and Test Loss')
      ax1.legend()
  ```

### Model Performance
- Training accuracy
- Test accuracy
- Confusion matrix analysis
- Error analysis

### Visualization
- Training progress plots
- Metric comparisons
- Performance analysis

---

## 7. Future Work
1. Model Improvements
   - Experiment with different architectures
   - Fine-tune hyperparameters
   - Ensemble methods

2. Feature Enhancements
   - Multi-class classification
   - Freshness level prediction
   - Real-time processing

3. Deployment
   - Mobile application
   - Web interface
   - API integration

---

## 8. Conclusion
- Successful implementation of food freshness detection
- Transfer learning proved effective
- Good balance of accuracy and efficiency
- Potential for real-world applications

---

## Thank You!
### Questions? 