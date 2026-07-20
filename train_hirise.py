# train_hirise_full.py - Full training on all 73,031 images
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import os
import time
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Add src directory to system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader_hirise import get_hirise_data_loaders
from vision_model import build_model, save_model, load_model

# ========== Configuration ==========
DATA_DIR = "data/HiRISE/map-proj-v3/"
LABELS_FILE = "data/HiRISE/labels-map-proj-v3.txt"
CLASSMAP_FILE = "data/HiRISE/landmarks_map-proj-v3_classmap.csv"
MODEL_SAVE_PATH = "models/trained_models/hirise_full_best_model.pth"
CHECKPOINT_PATH = "models/trained_models/hirise_full_checkpoint.pth"
HISTORY_PATH = "models/trained_models/hirise_full_history.json"

# Training parameters - Optimized for full dataset
BATCH_SIZE = 64          # زيادة حجم الدفعة لتسريع التدريب (إذا كانت الذاكرة تسمح)
EPOCHS = 50              # زيادة عدد العصور للحصول على دقة أفضل
LEARNING_RATE = 0.001
PATIENCE = 15            # زيادة الصبر للتدريب لفترة أطول
USE_AMP = False          # استخدم True إذا كان لديك GPU

# Data parameters - FULL DATASET
USE_SUBSET = False       # MUST BE False for full training

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===================================

# Create directories
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

print("="*60)
print("🚀 ASTROPILOT AI - FULL HiRISE Training")
print("="*60)
print(f"🔥 Using device: {DEVICE}")
print(f"📊 Batch size: {BATCH_SIZE}")
print(f"📈 Epochs: {EPOCHS}")
print(f"📉 Learning rate: {LEARNING_RATE}")
print(f"⏱️ Patience: {PATIENCE}")
print(f"📂 Dataset: FULL (73,031 images)")
print("="*60)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    """
    Train for one epoch with optional AMP (Automatic Mixed Precision).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    progress_bar = tqdm(loader, desc="Training", leave=False)
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with AMP if available
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100*correct/total:.2f}%',
            'time': f'{time.time()-start_time:.1f}s'
        })
    
    epoch_time = time.time() - start_time
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc, epoch_time


def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on validation/test set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Evaluating", leave=False)
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                'acc': f'{100*correct/total:.2f}%'
            })
    
    epoch_time = time.time() - start_time
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc, epoch_time


def save_checkpoint(model, optimizer, epoch, best_val_acc, history, le, filepath):
    """
    Save a checkpoint for resuming training.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
        'history': history,
        'classes': le.classes_ if le else None
    }, filepath)
    print(f"💾 Checkpoint saved at epoch {epoch}")


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load a checkpoint to resume training.
    """
    try:
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return (
        checkpoint['epoch'],
        checkpoint['best_val_acc'],
        checkpoint['history'],
        checkpoint.get('classes')
    )


def plot_training_history(history, save_path="training_history_full.png"):
    """
    Plot and save training history.
    """
    if not history['train_acc']:
        print("⚠️ No training history to plot")
        return
    
    epochs = range(1, len(history['train_acc']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    ax1.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Training and Validation Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax2.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training and Validation Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(MODEL_SAVE_PATH), save_path), dpi=150)
    print(f"📊 Training history plot saved to: {save_path}")
    # plt.show()


if __name__ == "__main__":
    start_time_total = time.time()
    
    # 1. Load Data - FULL DATASET
    print("\n📂 Loading FULL HiRISE V3 dataset (73,031 images)...")
    train_loader, val_loader, test_loader, label_encoder, num_classes = get_hirise_data_loaders(
        data_dir=DATA_DIR,
        labels_file=LABELS_FILE,
        classmap_file=CLASSMAP_FILE,
        batch_size=BATCH_SIZE
    )
    
    if train_loader is None:
        print("❌ Failed to load data. Please check file paths.")
        sys.exit(1)
    
    print(f"\n✅ Loaded {num_classes} classes")
    print(f"📊 Training batches: {len(train_loader)}")
    print(f"📊 Validation batches: {len(val_loader)}")
    print(f"📊 Test batches: {len(test_loader)}")
    
    # Display class distribution
    print("\n📊 Class Distribution in Training Set:")
    class_counts = train_loader.dataset.dataframe['label'].value_counts()
    for class_name, count in class_counts.items():
        class_name_str = str(class_name)
        try:
            class_idx = label_encoder.transform([class_name_str])[0] if class_name_str in label_encoder.classes_ else -1
        except:
            class_idx = -1
        print(f"  {class_name_str:20s}: {count:6d} samples (class {class_idx})")
    
    # 2. Build Model
    print("\n🧠 Building model...")
    model = build_model(num_classes, freeze_backbone=False)  # Fine-tune all layers
    model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")
    
    # 3. Setup training with better optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # AMP (Automatic Mixed Precision) for GPU
    scaler = torch.cuda.amp.GradScaler() if (USE_AMP and torch.cuda.is_available()) else None
    
    # Check for existing checkpoint
    start_epoch = 1
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'epoch_time': []}
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"\n💾 Found checkpoint: {CHECKPOINT_PATH}")
        resume = input("Resume training from checkpoint? (y/n): ").strip().lower()
        if resume == 'y':
            try:
                epoch, best_val_acc, history, classes = load_checkpoint(CHECKPOINT_PATH, model, optimizer)
                start_epoch = epoch + 1
                print(f"✅ Resuming from epoch {start_epoch}")
                print(f"📊 Best validation accuracy so far: {best_val_acc:.2f}%")
            except Exception as e:
                print(f"⚠️ Failed to load checkpoint: {e}")
                print("Starting fresh training...")
    
    # 4. Training Loop
    print(f"\n🚀 Starting FULL training from epoch {start_epoch} to {EPOCHS}...")
    print("⚠️ This will take a long time. Please be patient.")
    print("="*60)
    
    no_improve_count = 0
    
    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()
        print(f"\n📅 Epoch {epoch}/{EPOCHS}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc, train_time = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE, scaler
        )
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['epoch_time'].append(train_time)
        
        # Validate
        val_loss, val_acc, val_time = evaluate(model, val_loader, criterion, DEVICE)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Print results
        epoch_total_time = time.time() - epoch_start
        print(f"📈 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Time: {train_time:.1f}s")
        print(f"📉 Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}% | Time: {val_time:.1f}s")
        print(f"📊 LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"⏱️ Epoch total time: {epoch_total_time:.1f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, MODEL_SAVE_PATH, le=label_encoder)
            print(f"⭐ New best model saved! (Acc: {val_acc:.2f}%)")
            no_improve_count = 0
        else:
            no_improve_count += 1
            print(f"📊 No improvement for {no_improve_count} epochs")
        
        # Save checkpoint every epoch
        save_checkpoint(model, optimizer, epoch, best_val_acc, history, label_encoder, CHECKPOINT_PATH)
        
        # Early stopping
        if no_improve_count >= PATIENCE:
            print(f"⏹️ Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break
        
        # Save history every 5 epochs
        if epoch % 5 == 0:
            with open(HISTORY_PATH, 'w') as f:
                json.dump(history, f)
            print("💾 Training history saved")
    
    # 5. Final Evaluation
    print("\n" + "="*60)
    print("🎉 Training completed!")
    total_time = time.time() - start_time_total
    print(f"⏱️ Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Load best model for testing
    print("\n🧪 Evaluating on Test Set...")
    try:
        if os.path.exists(MODEL_SAVE_PATH):
            best_model, classes = load_model(MODEL_SAVE_PATH, num_classes, DEVICE)
            test_loss, test_acc, test_time = evaluate(best_model, test_loader, criterion, DEVICE)
            print(f"📊 Final Test Accuracy: {test_acc:.2f}%")
            print(f"📊 Final Test Loss: {test_loss:.4f}")
        else:
            print("⚠️ No best model found, using current model for testing")
            test_loss, test_acc, test_time = evaluate(model, test_loader, criterion, DEVICE)
            print(f"📊 Final Test Accuracy: {test_acc:.2f}%")
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        print("⚠️ Using current model for testing instead...")
        test_loss, test_acc, test_time = evaluate(model, test_loader, criterion, DEVICE)
        print(f"📊 Final Test Accuracy: {test_acc:.2f}%")
    
    # 6. Plot Training History
    try:
        plot_training_history(history)
    except Exception as e:
        print(f"⚠️ Could not plot history: {e}")
    
    # 7. Summary
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)
    print(f"  Device:         {DEVICE}")
    print(f"  Total Epochs:   {epoch}")
    print(f"  Best Val Acc:   {best_val_acc:.2f}%")
    print(f"  Test Acc:       {test_acc:.2f}%")
    print(f"  Total Time:     {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"  Model saved:    {MODEL_SAVE_PATH}")
    print(f"  Checkpoint:     {CHECKPOINT_PATH}")
    print(f"  History:        {HISTORY_PATH}")
    print("="*60)
    
    # 8. Show label mapping
    print("\n📝 Label Mapping:")
    for idx, class_name in enumerate(label_encoder.classes_):
        print(f"  Class {idx}: {class_name}")
    
    print("\n✅ Training completed successfully!")
    print("🚀 You can now use the model for predictions:")
    print(f"   python predict_hirise.py path/to/image.jpg")