import torch
import time

def train_epoch(model, loader, criterion, optimizer, device):
    """Performs a single training epoch."""
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)

def evaluate_model(model, loader, device, criterion=None):
    """Evaluates the model's accuracy and optionally loss."""
    model.eval()
    correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
    
    accuracy = correct / len(loader.dataset)
    avg_loss = total_loss / len(loader.dataset) if criterion else 0.0
    return accuracy, avg_loss

def train_model(model, train_loader, validation_loader, criterion, optimizer, scheduler, num_epochs, device):
    """Main training loop that returns the training history."""
    history = {'train_loss': [], 'val_acc': []}
    print(f"Starting training for {num_epochs} epochs.")

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_acc, _ = evaluate_model(model, validation_loader, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Duration: {time.time() - start_time:.2f}s")
    
    print("Training finished.")
    return history