# src/vision_model.py
import torch
import torch.nn as nn
from torchvision import models
import os


def build_model(num_classes, freeze_backbone=True):
    """
    Build a ResNet18 model with ImageNet pre-trained weights.
    If freeze_backbone=True, the convolutional base is frozen for faster training.
    """
    # Load pre-trained weights
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze backbone layers (optional, helps with small datasets)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully-connected layer to match our number of classes
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),  # Regularization to prevent overfitting
        nn.Linear(num_features, num_classes)
    )
    
    return model


def save_model(model, save_path, le=None):
    """
    Save the trained model weights along with the LabelEncoder classes.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': le.classes_ if le else None
    }, save_path)
    print(f"✅ Model saved successfully at: {save_path}")


def load_model(model_path, num_classes, device='cpu'):
    """
    Load a saved model for inference.
    """
    model = build_model(num_classes, freeze_backbone=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    classes = checkpoint.get('classes')
    return model, classes


def predict_image(model, image_tensor, device='cpu'):
    """
    Run inference on a single image tensor.
    Returns the predicted class index and probabilities.
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        return predicted.item(), probabilities.cpu().numpy()[0]