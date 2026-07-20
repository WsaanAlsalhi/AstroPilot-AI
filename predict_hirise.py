# predict_hirise.py - Predict terrain from an image
import torch
import sys
import os
from PIL import Image
from torchvision import transforms
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from vision_model import load_model
from decision_engine import DecisionEngine
from gpt_agent import GPTAgent

# ========== Configuration ==========
MODEL_PATH = "models/trained_models/hirise_fast_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 8  # تغيير حسب عدد الفئات في بياناتك
# ===================================

# Class names (من ملف classmap)
CLASS_NAMES = [
    'unknown',      # 0
    'crater',       # 1
    'slope streak', # 2
    'bright dune',  # 3
    'swiss cheese', # 4
    'dark dune',    # 5
    'spider',       # 6
    'impact ejecta' # 7
]

def predict_image(image_path):
    """
    Predict terrain from a single image.
    """
    print(f"📸 Loading image: {image_path}")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None
    
    # Load model
    print("🧠 Loading model...")
    try:
        model, classes = load_model(MODEL_PATH, NUM_CLASSES, DEVICE)
        if classes is not None:
            print(f"✅ Loaded classes: {classes}")
    except FileNotFoundError:
        print(f"❌ Model not found at: {MODEL_PATH}")
        print("Please train the model first using: python train_hirise.py")
        return None
    
    # Predict
    print("🔮 Predicting...")
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(outputs, 1).item()
    
    # Get class name
    if classes is not None and predicted_idx < len(classes):
        predicted_class = classes[predicted_idx]
    else:
        predicted_class = CLASS_NAMES[predicted_idx] if predicted_idx < len(CLASS_NAMES) else f"Class {predicted_idx}"
    
    confidence = probabilities[0][predicted_idx].item() * 100
    
    print(f"\n✅ Predicted Terrain: {predicted_class}")
    print(f"📊 Confidence: {confidence:.2f}%")
    
    # Show top 3 predictions
    print("\n📊 Top 3 Predictions:")
    top3 = torch.topk(probabilities[0], 3)
    for i, (prob, idx) in enumerate(zip(top3.values, top3.indices)):
        idx = idx.item()
        class_name = classes[idx] if classes is not None and idx < len(classes) else CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class {idx}"
        print(f"  {i+1}. {class_name}: {prob.item()*100:.2f}%")
    
    # Decision Engine
    print("\n" + "="*50)
    print("🤖 ASTROPILOT AI - MISSION DECISION")
    print("="*50)
    
    decision_engine = DecisionEngine()
    risk_assessment = decision_engine.assess_risk(predicted_class, battery_level=75, communication_delay=10)
    
    print(f"\n📊 Risk Assessment:")
    print(f"  Terrain: {risk_assessment['terrain_label']}")
    print(f"  Terrain Risk: {risk_assessment['terrain_risk']:.2f}")
    print(f"  Battery Risk: {risk_assessment['battery_risk']:.2f}")
    print(f"  Communication Risk: {risk_assessment['comm_risk']:.2f}")
    print(f"  Overall Risk: {risk_assessment['overall_risk']:.2f}")
    
    print(f"\n🎯 Decision: {risk_assessment['decision']}")
    print(f"📝 Action: {risk_assessment['action']}")
    print(f"📊 Confidence: {risk_assessment['confidence']}")
    
    # GPT Agent Report
    print("\n" + "="*50)
    print("📡 AI MISSION REPORT")
    print("="*50)
    
    gpt_agent = GPTAgent()
    report = gpt_agent.generate_mission_report(risk_assessment, battery=75, comm_delay=10)
    print(report)
    
    return predicted_class, risk_assessment


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default: use a sample image if available
        image_path = "data/HiRISE/map-proj-v3/sample.jpg"
        if not os.path.exists(image_path):
            print("Usage: python predict_hirise.py <image_path>")
            print(f"Example: python predict_hirise.py data/HiRISE/map-proj-v3/your_image.jpg")
            sys.exit(1)
    
    predict_image(image_path)