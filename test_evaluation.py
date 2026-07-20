# test_evaluation.py - Evaluate trained model on test set (FULLY FIXED VERSION)
import torch
import torch.nn as nn
import sys
import os
import time
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add src directory to system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader_hirise import get_hirise_data_loaders
from vision_model import build_model

# ========== Configuration ==========
DATA_DIR = "data/HiRISE/map-proj-v3/"
LABELS_FILE = "data/HiRISE/labels-map-proj-v3.txt"
CLASSMAP_FILE = "data/HiRISE/landmarks_map-proj-v3_classmap.csv"
MODEL_PATH = "models/trained_models/hirise_full_best_model.pth"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===================================

print("="*60)
print("🧪 ASTROPILOT AI - Model Evaluation")
print("="*60)
print(f"🔥 Using device: {DEVICE}")
print(f"📊 Batch size: {BATCH_SIZE}")
print("="*60)


def load_model_direct(model_path, num_classes, device='cpu'):
    """Load model with weights_only=False for PyTorch 2.6+"""
    print(f"🧠 Loading model from: {model_path}")
    model = build_model(num_classes, freeze_backbone=False)
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    classes = checkpoint.get('classes')
    return model, classes


def evaluate_model(model, loader, criterion, device, class_names=None):
    """Comprehensive evaluation of the model on a dataset."""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    total = 0
    start_time = time.time()
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Evaluating")
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            current_acc = accuracy_score(all_labels, all_predictions) * 100
            progress_bar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{current_acc:.2f}%'
            })
    
    epoch_time = time.time() - start_time
    epoch_loss = running_loss / total
    epoch_acc = accuracy_score(all_labels, all_predictions) * 100
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probabilities),
        'time': epoch_time
    }


def plot_confusion_matrix(labels, predictions, class_names, save_path="confusion_matrix.png"):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path_full = os.path.join(os.path.dirname(MODEL_PATH), save_path)
    plt.savefig(save_path_full, dpi=150)
    print(f"📊 Confusion matrix saved to: {save_path_full}")


def plot_probability_distribution(probabilities, labels, class_names, save_path="probability_distribution.png"):
    """Plot probability distribution for correct and incorrect predictions."""
    max_probs = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    correct = (predictions == labels)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.hist(max_probs[correct], bins=50, alpha=0.7, color='green', label='Correct')
    ax1.hist(max_probs[~correct], bins=50, alpha=0.7, color='red', label='Incorrect')
    ax1.set_xlabel('Confidence Probability')
    ax1.set_ylabel('Count')
    ax1.set_title('Confidence Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    class_acc = []
    for i in range(len(class_names)):
        mask = (labels == i)
        if mask.sum() > 0:
            acc = (predictions[mask] == labels[mask]).mean() * 100
        else:
            acc = 0
        class_acc.append(acc)
    
    bars = ax2.bar(range(len(class_names)), class_acc, color='skyblue')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy per Class')
    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    for bar, acc in zip(bars, class_acc):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_path_full = os.path.join(os.path.dirname(MODEL_PATH), save_path)
    plt.savefig(save_path_full, dpi=150)
    print(f"📊 Probability distribution saved to: {save_path_full}")


def print_detailed_report(results, class_names):
    """Print detailed classification report."""
    print("\n" + "="*60)
    print("📊 DETAILED CLASSIFICATION REPORT")
    print("="*60)
    
    print(f"\n📈 Overall Metrics:")
    print(f"  Accuracy:  {results['accuracy']:.2f}%")
    print(f"  Loss:      {results['loss']:.4f}")
    print(f"  Time:      {results['time']:.2f}s")
    
    print("\n📊 Per-Class Metrics:")
    print("-" * 70)
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    
    report = classification_report(results['labels'], results['predictions'],
                                   target_names=class_names, output_dict=True)
    
    for class_name in class_names:
        if class_name in report:
            metrics = report[class_name]
            print(f"{class_name:<20} {metrics['precision']*100:>6.2f}%    "
                  f"{metrics['recall']*100:>6.2f}%    {metrics['f1-score']*100:>6.2f}%    "
                  f"{int(metrics['support']):>6d}")
    
    print("-" * 70)
    print(f"{'Macro Avg':<20} {report['macro avg']['precision']*100:>6.2f}%    "
          f"{report['macro avg']['recall']*100:>6.2f}%    {report['macro avg']['f1-score']*100:>6.2f}%")
    print(f"{'Weighted Avg':<20} {report['weighted avg']['precision']*100:>6.2f}%    "
          f"{report['weighted avg']['recall']*100:>6.2f}%    {report['weighted avg']['f1-score']*100:>6.2f}%")
    print("-" * 70)


def save_results(results, class_names, save_path="evaluation_results.json"):
    """Save evaluation results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    results_to_save = {
        'accuracy': float(results['accuracy']),
        'loss': float(results['loss']),
        'time': float(results['time']),
        'class_names': class_names,
        'predictions': results['predictions'].tolist(),
        'labels': results['labels'].tolist(),
    }
    
    # Add per-class metrics
    report = classification_report(results['labels'], results['predictions'],
                                   target_names=class_names, output_dict=True)
    results_to_save['per_class_metrics'] = {}
    for class_name in class_names:
        if class_name in report:
            results_to_save['per_class_metrics'][class_name] = {
                'precision': float(report[class_name]['precision']),
                'recall': float(report[class_name]['recall']),
                'f1-score': float(report[class_name]['f1-score']),
                'support': int(report[class_name]['support'])
            }
    
    results_to_save['macro_avg'] = {
        'precision': float(report['macro avg']['precision']),
        'recall': float(report['macro avg']['recall']),
        'f1-score': float(report['macro avg']['f1-score'])
    }
    results_to_save['weighted_avg'] = {
        'precision': float(report['weighted avg']['precision']),
        'recall': float(report['weighted avg']['recall']),
        'f1-score': float(report['weighted avg']['f1-score'])
    }
    
    save_path_full = os.path.join(os.path.dirname(MODEL_PATH), save_path)
    with open(save_path_full, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"📊 Results saved to: {save_path_full}")


def main():
    """Main evaluation function."""
    # 1. Load Data
    print("\n📂 Loading data...")
    train_loader, val_loader, test_loader, label_encoder, num_classes = get_hirise_data_loaders(
        data_dir=DATA_DIR,
        labels_file=LABELS_FILE,
        classmap_file=CLASSMAP_FILE,
        batch_size=BATCH_SIZE
    )
    
    if test_loader is None:
        print("❌ Failed to load data. Please check file paths.")
        sys.exit(1)
    
    class_names = label_encoder.classes_
    print(f"\n✅ Loaded {num_classes} classes")
    print(f"📊 Test batches: {len(test_loader)}")
    print(f"📊 Total test samples: {len(test_loader.dataset)}")
    
    # 2. Load Model
    print(f"\n🧠 Loading model from: {MODEL_PATH}")
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model not found at: {MODEL_PATH}")
            print("\n📂 Available models in directory:")
            model_dir = os.path.dirname(MODEL_PATH)
            if os.path.exists(model_dir):
                for f in os.listdir(model_dir):
                    if f.endswith('.pth'):
                        print(f"  - {f}")
            sys.exit(1)
        
        model, classes = load_model_direct(MODEL_PATH, num_classes, DEVICE)
        print("✅ Model loaded successfully!")
        
        if classes is not None:
            print(f"📊 Model classes: {classes}")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # 3. Evaluate on Test Set
    print("\n🧪 Evaluating on Test Set...")
    criterion = nn.CrossEntropyLoss()
    results = evaluate_model(model, test_loader, criterion, DEVICE, class_names)
    
    # 4. Print Detailed Report
    print_detailed_report(results, class_names)
    
    # 5. Save Results
    save_results(results, class_names)
    
    # 6. Plot Confusion Matrix
    try:
        print("\n📊 Generating confusion matrix...")
        plot_confusion_matrix(results['labels'], results['predictions'], class_names)
    except Exception as e:
        print(f"⚠️ Could not plot confusion matrix: {e}")
    
    # 7. Plot Probability Distribution
    try:
        print("\n📊 Generating probability distribution...")
        plot_probability_distribution(results['probabilities'], results['labels'], class_names)
    except Exception as e:
        print(f"⚠️ Could not plot probability distribution: {e}")
    
    # 8. Summary
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETED")
    print("="*60)
    print(f"  Test Accuracy:  {results['accuracy']:.2f}%")
    print(f"  Test Loss:      {results['loss']:.4f}")
    print(f"  Test Time:      {results['time']:.2f}s")
    print(f"  Total Samples:  {len(results['labels'])}")
    print("="*60)
    
    # 9. Show sample predictions
    print("\n🔍 Sample Predictions (10 random samples):")
    print("-" * 70)
    indices = np.random.choice(len(results['labels']), min(10, len(results['labels'])), replace=False)
    for i in indices:
        true_label = class_names[results['labels'][i]]
        pred_label = class_names[results['predictions'][i]]
        prob = results['probabilities'][i][results['predictions'][i]] * 100
        status = "✅" if results['labels'][i] == results['predictions'][i] else "❌"
        print(f"  {status} Sample {i:4d}: True: {true_label:15s} | Pred: {pred_label:15s} | Conf: {prob:6.2f}%")
    print("-" * 70)
    
    print("\n📋 Model Information:")
    print(f"  Model Path: {MODEL_PATH}")
    print(f"  Device: {DEVICE}")
    print(f"  Classes: {len(class_names)}")


if __name__ == "__main__":
    main()