import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torch

def init_weights(module, strategy='he'):
    """Helper function to initialize the weights of a model."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        if strategy.lower() == 'he':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif strategy.lower() == 'xavier':
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def analyze_performance(model, loader, class_names, device):
    """Generates and displays a classification report and confusion matrix."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(device))
            preds = torch.max(outputs, 1)[1]
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.show()

def plot_comparison(vgg_history, resnet_history):
    """Plots the performance comparison between VGG and ResNet."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    ax1.plot(vgg_history['val_acc'], label='VGG-16 Val Accuracy', color='blue', linestyle='--')
    ax1.plot(resnet_history['val_acc'], label='ResNet-18 Val Accuracy', color='green', linestyle='-')
    ax1.set_title('Validation Accuracy vs. Epochs')
    ax1.set_xlabel('Epochs'); ax1.set_ylabel('Accuracy'); ax1.legend(); ax1.grid(True)

    ax2.plot(vgg_history['train_loss'], label='VGG-16 Train Loss', color='red', linestyle='--')
    ax2.plot(resnet_history['train_loss'], label='ResNet-18 Train Loss', color='orange', linestyle='-')
    ax2.set_title('Training Loss vs. Epochs')
    ax2.set_xlabel('Epochs'); ax2.set_ylabel('Loss'); ax2.legend(); ax2.grid(True)
    plt.show()