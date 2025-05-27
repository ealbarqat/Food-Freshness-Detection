import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import argparse
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns

class FoodDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['fresh', 'rotten']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        # Load fresh images
        fresh_dir = os.path.join(root_dir, 'fresh')
        for img_name in os.listdir(fresh_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                self.images.append(os.path.join(fresh_dir, img_name))
                self.labels.append(0)  # 0 for fresh
        
        # Load rotten images
        rotten_dir = os.path.join(root_dir, 'rotten')
        for img_name in os.listdir(rotten_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                self.images.append(os.path.join(rotten_dir, img_name))
                self.labels.append(1)  # 1 for rotten

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class FoodFreshnessModel(nn.Module):
    def __init__(self):
        super(FoodFreshnessModel, self).__init__()
        # Load pretrained ResNet50
        self.model = models.resnet50(pretrained=True)
        
        # Freeze all layers except the last few
        for param in list(self.model.parameters())[:-20]:
            param.requires_grad = False
            
        # Modify the final layer for binary classification
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def plot_metrics(metrics_history):
    """Plot training metrics over epochs"""
    epochs = range(1, len(metrics_history['train_loss']) + 1)
    
    # Create a figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot losses
    ax1.plot(epochs, metrics_history['train_loss'], label='Train Loss')
    ax1.plot(epochs, metrics_history['test_loss'], label='Test Loss')
    ax1.set_title('Training and Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(epochs, metrics_history['train_acc'], label='Train Accuracy')
    ax2.plot(epochs, metrics_history['test_acc'], label='Test Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Plot precision and recall
    ax3.plot(epochs, metrics_history['test_prec'], label='Test Precision')
    ax3.plot(epochs, metrics_history['test_rec'], label='Test Recall')
    ax3.set_title('Test Precision and Recall')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.legend()
    ax3.grid(True)
    
    # Plot F1 score
    ax4.plot(epochs, metrics_history['test_f1'], label='Test F1 Score')
    ax4.set_title('Test F1 Score')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('F1 Score')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def plot_confusion_matrix(cm, classes=['Fresh', 'Rotten']):
    """Plot and save confusion matrix visualization"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.float().to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        preds = (outputs.squeeze() > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return running_loss / len(train_loader), accuracy, precision, recall, f1

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.float().to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            
            running_loss += loss.item()
            
            preds = (outputs.squeeze() > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm)
    
    return running_loss / len(test_loader), cm, accuracy, precision, recall, f1

def predict_image(model, image_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(image)
        prediction = (output.squeeze() > 0.5).float()
        
    return "Fresh" if prediction.item() == 0 else "Rotten"

def main():
    parser = argparse.ArgumentParser(description='Food Freshness Detection')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                      help='Mode: train or predict')
    parser.add_argument('--image_path', type=str,
                      help='Path to image for prediction')
    args = parser.parse_args()

    # Set device
    device = torch.device('cpu')  # Force CPU usage
    print(f"Using device: {device}")

    if args.mode == 'train':
        # Data transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create dataset and dataloader
        dataset = FoodDataset('.', transform=transform)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        # Smaller batch size for CPU
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

        # Initialize model, criterion, and optimizer
        model = FoodFreshnessModel().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

        # Training loop with early stopping
        num_epochs = 30  # Increased epochs
        best_loss = float('inf')
        patience = 10  # Increased patience
        patience_counter = 0
        
        # Initialize metrics history
        metrics_history = {
            'train_loss': [], 'test_loss': [],
            'train_acc': [], 'test_acc': [],
            'test_prec': [], 'test_rec': [],
            'test_f1': []
        }
        
        print("\nStarting training...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_acc, train_prec, train_rec, train_f1 = train_model(
                model, train_loader, criterion, optimizer, device)
            
            # Evaluate
            test_loss, cm, test_acc, test_prec, test_rec, test_f1 = evaluate_model(
                model, test_loader, criterion, device)
            
            # Update learning rate
            scheduler.step(test_loss)
            
            # Store metrics
            metrics_history['train_loss'].append(train_loss)
            metrics_history['test_loss'].append(test_loss)
            metrics_history['train_acc'].append(train_acc)
            metrics_history['test_acc'].append(test_acc)
            metrics_history['test_prec'].append(test_prec)
            metrics_history['test_rec'].append(test_rec)
            metrics_history['test_f1'].append(test_f1)
            
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
            print(f"Test Precision: {test_prec:.4f}, Test Recall: {test_rec:.4f}, Test F1: {test_f1:.4f}")
            print("Confusion Matrix:")
            print(cm)
            
            # Early stopping
            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'model.pth')
                print("New best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/60:.2f} minutes")
        print("Best model saved as 'model.pth'")
        
        # Plot and save metrics
        plot_metrics(metrics_history)
        print("Training metrics plot saved as 'training_metrics.png'")

    elif args.mode == 'predict':
        if not args.image_path:
            print("Error: Please provide an image path for prediction")
            return

        # Load the model
        model = FoodFreshnessModel().to(device)
        model.load_state_dict(torch.load('model.pth'))
        
        # Make prediction
        result = predict_image(model, args.image_path, device)
        print(f"\nPrediction: {result}")

if __name__ == '__main__':
    main() 