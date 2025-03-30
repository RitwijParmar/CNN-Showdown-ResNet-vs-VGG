import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import data_setup
import models
import engine
import utils

# --- CONFIGURATION ---
DATASET_ROOT = "/path/to/your/dataset_folder" # IMPORTANT: Update this path
BATCH_SIZE = 128
NUM_EPOCHS = 10
SAVE_PATH = "best_model.pth"

# --- MAIN SCRIPT ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get DataLoaders
    train_loader, val_loader, test_loader, class_names = data_setup.get_dataloaders(
        root_dir=DATASET_ROOT, 
        batch_size=BATCH_SIZE
    )

    # --- VGG-16 Training ---
    print("\n--- Training VGG-16 ---")
    vgg_model = models.VggNet(num_classes=len(class_names)).to(device)
    vgg_model.apply(utils.init_weights)
    optimizer_vgg = optim.AdamW(vgg_model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler_vgg = lr_scheduler.MultiStepLR(optimizer_vgg, milestones=[8], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    vgg_history = engine.train_model(
        vgg_model, train_loader, val_loader, criterion, optimizer_vgg, scheduler_vgg, NUM_EPOCHS, device
    )

    # --- ResNet-18 Training ---
    print("\n--- Training ResNet-18 ---")
    resnet_model = models.ResNet18(num_classes=len(class_names)).to(device)
    resnet_model.apply(utils.init_weights)
    optimizer_resnet = optim.Adam(resnet_model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler_resnet = lr_scheduler.MultiStepLR(optimizer_resnet, milestones=[8], gamma=0.1)

    resnet_history = engine.train_model(
        resnet_model, train_loader, val_loader, criterion, optimizer_resnet, scheduler_resnet, NUM_EPOCHS, device
    )

    # --- EVALUATION & ANALYSIS ---
    print("\n--- VGG-16 Test Set Evaluation ---")
    utils.analyze_performance(vgg_model, test_loader, class_names, device)
    
    print("\n--- ResNet-18 Test Set Evaluation ---")
    utils.analyze_performance(resnet_model, test_loader, class_names, device)
    
    print("\n--- VGG-16 vs. ResNet-18 Performance Comparison ---")
    utils.plot_comparison(vgg_history, resnet_history)

    # Determine and save the best model
    vgg_test_acc, _ = engine.evaluate_model(vgg_model, test_loader, device)
    resnet_test_acc, _ = engine.evaluate_model(resnet_model, test_loader, device)
    
    if resnet_test_acc > vgg_test_acc:
        print(f"\nResNet-18 performed best with test accuracy: {resnet_test_acc:.4f}")
        torch.save(resnet_model.state_dict(), SAVE_PATH)
    else:
        print(f"\nVGG-16 performed best with test accuracy: {vgg_test_acc:.4f}")
        torch.save(vgg_model.state_dict(), SAVE_PATH)
    
    print(f"Best model weights saved to: {SAVE_PATH}")