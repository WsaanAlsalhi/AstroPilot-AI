# predict.py
import sys
import os
import torch

# Add src directory to system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import get_single_image_loader
from vision_model import load_model, predict_image
from decision_engine import DecisionEngine
from gpt_agent import GPTAgent

# ========== Configuration ==========
MODEL_PATH = "models/trained_models/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 7  # Adjust this based on your dataset
# ===================================


def predict_terrain(image_path):
    """
    Predict terrain from a single image.
    """
    print(f"📂 Loading image: {image_path}")
    
    # Load image
    image_tensor = get_single_image_loader(image_path)
    if image_tensor is None:
        print("❌ Failed to load image")
        return None
    
    # Load model
    print("🧠 Loading model...")
    model, classes = load_model(MODEL_PATH, NUM_CLASSES, DEVICE)
    
    if classes is None:
        print("⚠️ Warning: Classes not found in model checkpoint")
        classes = ['bright dune', 'crater', 'dark dune', 'impact ejecta', 
                   'slope streak', 'spider', 'swiss cheese']
    
    # Predict
    print("🔮 Predicting...")
    predicted_idx, probabilities = predict_image(model, image_tensor, DEVICE)
    predicted_class = classes[predicted_idx] if classes else f"Class {predicted_idx}"
    
    print(f"\n✅ Predicted Terrain: {predicted_class}")
    print(f"📊 Confidence: {probabilities[predicted_idx]*100:.2f}%")
    
    # Show top 3 predictions
    print("\n📊 Top 3 Predictions:")
    sorted_indices = probabilities.argsort()[-3:][::-1]
    for idx in sorted_indices:
        class_name = classes[idx] if classes else f"Class {idx}"
        print(f"  {class_name}: {probabilities[idx]*100:.2f}%")
    
    # Decision Engine
    print("\n" + "="*50)
    print("🤖 ASTROPILOT AI - MISSION DECISION")
    print("="*50)
    
    decision_engine = DecisionEngine()
    risk_assessment = decision_engine.assess_risk(predicted_class, battery_level=70, communication_delay=10)
    
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
    report = gpt_agent.generate_mission_report(risk_assessment, battery=70, comm_delay=10)
    print(report)
    
    explanation = gpt_agent.explain_decision(risk_assessment)
    print(explanation)
    
    return predicted_class, risk_assessment


if __name__ == "__main__":
    # Example usage
    image_path = "data/HiRISE/images/sample_image.jpg"
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("Please place an image at: data/HiRISE/images/sample_image.jpg")
        print("Or modify the image_path variable.")
    else:
        predict_terrain(image_path)