# src/utils.py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
from collections import Counter


def imshow(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], title=None):
    """
    Display a batch of images (denormalize and convert to numpy).
    """
    # Denormalize
    img_tensor = img_tensor.clone().detach().cpu()
    for t, m, s in zip(img_tensor, mean, std):
        t.mul_(s).add_(m)
    # Clamp to [0,1]
    img_tensor = torch.clamp(img_tensor, 0, 1)
    
    # Convert to grid and plot
    grid = make_grid(img_tensor, nrow=4)
    np_img = grid.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(10, 8))
    plt.imshow(np_img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


def get_class_distribution(dataset, label_encoder):
    """
    Print the number of samples per class in a dataset.
    """
    labels = [item[1] for item in dataset]
    if isinstance(labels[0], str):
        labels = label_encoder.transform(labels)
    counter = Counter(labels)
    total = sum(counter.values())
    print("\n📊 Class Distribution:")
    for class_idx, count in counter.items():
        class_name = label_encoder.classes_[class_idx]
        print(f"  {class_name}: {count} ({count/total*100:.1f}%)")


def print_training_summary(history):
    """
    Print a summary of training history.
    """
    print("\n📈 Training Summary:")
    print(f"  Final Train Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"  Final Val Accuracy: {history['val_acc'][-1]:.2f}%")
    print(f"  Best Val Accuracy: {max(history['val_acc']):.2f}%")
    print(f"  Best Val Loss: {min(history['val_loss']):.4f}")