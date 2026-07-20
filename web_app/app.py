# web_app/app.py - Flask web application for AstroPilot AI
import os
import sys
import torch
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision import transforms
import numpy as np
import uuid

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vision_model import build_model
from src.decision_engine import DecisionEngine
from src.gpt_agent import GPTAgent

# ========== Configuration ==========
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'web_app/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Model configuration
MODEL_PATH = "models/trained_models/hirise_full_best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 8

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model (lazy loading)
model = None
classes = None

def get_model():
    """Lazy load model with weights_only=False for PyTorch 2.6+"""
    global model, classes
    if model is None:
        print("🧠 Loading model...")
        try:
            # Try with weights_only=False (PyTorch 2.6+)
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            model = build_model(NUM_CLASSES, freeze_backbone=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(DEVICE)
            model.eval()
            classes = checkpoint.get('classes')
            print(f"✅ Model loaded on {DEVICE}")
        except TypeError:
            # Fallback for older PyTorch versions
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            model = build_model(NUM_CLASSES, freeze_backbone=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(DEVICE)
            model.eval()
            classes = checkpoint.get('classes')
            print(f"✅ Model loaded on {DEVICE} (fallback mode)")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    return model, classes

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_image(image_path, battery=75, comm_delay=10, mission_priority='science'):
    """Predict terrain from image with enhanced analysis"""
    model, classes = get_model()
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    except Exception as e:
        return {'error': f'Error loading image: {str(e)}'}
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(outputs, 1).item()
    
    # Get class name
    if classes is not None and predicted_idx < len(classes):
        predicted_class = classes[predicted_idx]
    else:
        predicted_class = f"Class {predicted_idx}"
    
    confidence = probabilities[0][predicted_idx].item() * 100
    
    # Get top 5 predictions
    top_indices = torch.topk(probabilities[0], 5)
    top_predictions = []
    for prob, idx in zip(top_indices.values, top_indices.indices):
        idx = idx.item()
        class_name = classes[idx] if classes is not None and idx < len(classes) else f"Class {idx}"
        top_predictions.append({
            'class': class_name,
            'probability': prob.item() * 100
        })
    
    # Decision Engine with enhanced analysis
    decision_engine = DecisionEngine()
    risk_assessment = decision_engine.assess_risk(
        predicted_class, 
        battery_level=battery, 
        communication_delay=comm_delay,
        mission_priority=mission_priority
    )
    
    # GPT Report
    gpt_agent = GPTAgent()
    report = gpt_agent.generate_mission_report(
        risk_assessment, 
        battery=battery, 
        comm_delay=comm_delay
    )
    
    return {
        'terrain': predicted_class,
        'confidence': confidence,
        'top_predictions': top_predictions,
        'risk_assessment': risk_assessment,
        'report': report,
        'battery': battery,
        'comm_delay': comm_delay,
        'mission_priority': mission_priority
    }

# ========== Routes ==========

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    # Check if file is present
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Please upload JPG, PNG, or GIF.'}), 400
    
    # Get parameters
    battery = int(request.form.get('battery', 75))
    comm_delay = int(request.form.get('comm_delay', 10))
    mission_priority = request.form.get('mission_priority', 'science')
    
    # Save file
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)
    
    try:
        # Predict with enhanced analysis
        result = predict_image(filepath, battery, comm_delay, mission_priority)
        
        if 'error' in result:
            return jsonify(result), 500
        
        # Add image URL
        result['image_url'] = f"/uploads/{unique_filename}"
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up uploaded file (optional)
        # os.remove(filepath)
        pass

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded images"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'device': str(DEVICE)})

if __name__ == '__main__':
    # Pre-load model
    print("🚀 Starting AstroPilot AI Web Server...")
    try:
        get_model()
        print("🌐 Server running at: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")