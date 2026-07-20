# 🚀 AstroPilot AI

## Autonomous Mission Decision AI for Space Exploration

<<<<<<< HEAD
## 🌌 Project Overview

AstroPilot AI is an AI-powered autonomous mission assistant designed to support decision-making for space exploration missions involving **Rovers, Satellites, and Landers**.
=======
[![OpenAI Build Week](https://img.shields.io/badge/OpenAI-Build%20Week-blue)](https://openai.devpost.com)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-2.2+-lightblue)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An AI-powered autonomous mission assistant for space exploration using Computer Vision and GPT-5.6**

---

## 🌌 Project Overview

**AstroPilot AI** is an AI-powered autonomous mission assistant designed to support decision-making for space exploration missions involving **Rovers, Satellites, and Landers**.
>>>>>>> 22ab4f3ab5eab8abd76eee86282351aa1f68dd11

The system analyzes planetary environments using real space imagery from NASA's **Mars Orbital Image HiRISE Dataset**, extracts terrain characteristics, evaluates mission risks, and generates intelligent recommendations using GPT-5.6.

Instead of relying on human operators for every decision, AstroPilot AI acts as a virtual mission officer that helps autonomous systems choose safer and more efficient actions.

---

<<<<<<< HEAD
# 💡 Inspiration

Space missions operate in extremely challenging environments where communication delays, limited energy resources, and unpredictable terrain can affect mission success.

Inspired by autonomous Mars exploration missions, we wanted to create an AI system capable of assisting spacecraft in making faster and smarter decisions without requiring continuous human intervention.

AstroPilot AI combines computer vision, decision intelligence, and large language models to create an intelligent mission assistant.

---

# ⚙️ What It Does

AstroPilot AI performs the following tasks:

### 🛰️ Terrain Analysis

* Processes planetary images from NASA HiRISE datasets.
* Identifies terrain characteristics.
* Detects potential hazards.

### 🤖 Autonomous Decision Making

Evaluates mission conditions such as:

* Terrain difficulty
* Safety risks
* Battery limitations
* Communication delay

Then generates recommendations:

* Continue mission
* Change route
* Enter safe mode
* Avoid dangerous areas

### 🧠 AI Mission Assistant

Using GPT-5.6, AstroPilot AI explains decisions in a human-readable mission report.

Example:

```
Mission Status:
HIGH RISK

Reason:
The selected path contains unstable rocky terrain
with limited recovery possibilities.

Recommendation:
Navigate through Route B.
```

---

# 🏗️ System Architecture

```
                User Dashboard
                       |
                       |
                Mission Input
=======
## 💡 Inspiration

Space missions operate in extremely challenging environments where communication delays, limited energy resources, and unpredictable terrain can affect mission success.

Inspired by autonomous Mars exploration missions, we created an AI system capable of assisting spacecraft in making faster and smarter decisions without requiring continuous human intervention.

---

## ⚙️ What It Does

### 🛰️ Terrain Analysis
- Processes planetary images from NASA HiRISE datasets
- Identifies terrain characteristics (rocky, sandy, crater, etc.)
- Detects potential hazards
- Achieves **87.77% accuracy** on 73,031 images

### 🤖 Autonomous Decision Making
Evaluates mission conditions such as:
- Terrain difficulty
- Safety risks
- Battery limitations
- Communication delay

Generates recommendations:
- Continue mission
- Change route
- Enter safe mode
- Avoid dangerous areas

### 🧠 AI Mission Assistant
Using **GPT-5.6**, AstroPilot AI explains decisions in a human-readable mission report with:
- Detailed terrain descriptions
- Risk assessment breakdown
- Actionable recommendations
- Natural language explanations

### 📡 Web Interface
- Interactive dashboard for uploading and analyzing images
- Real-time terrain classification and risk assessment
- GPT-5.6 generated mission reports
- Visual feedback with mission status

---

## 🏗️ System Architecture

```
                User Dashboard (Web Interface)
                       |
                       ↓
                Mission Input (Image + Parameters)
>>>>>>> 22ab4f3ab5eab8abd76eee86282351aa1f68dd11
                       |
                       ↓
              AstroPilot AI Core
                       |
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
  Computer Vision   Decision      GPT-5.6
     Module         Engine        Agent
        ↓              ↓              ↓
 Terrain Features  Mission Plan  AI Report
<<<<<<< HEAD
=======
        ↓              ↓              ↓
   ──────────────────────────────────────
        ↓              ↓              ↓
          Risk Assessment & Recommendations
                       |
                       ↓
                Mission Decision
>>>>>>> 22ab4f3ab5eab8abd76eee86282351aa1f68dd11
```

---

<<<<<<< HEAD
# 📂 Project Structure

```
AstroPilot-AI/

│
├── data/
│   └── HiRISE/
│       ├── images/
│       └── labels.csv
│
├── src/
│   │
│   ├── data_loader.py
│   │
│   ├── vision_model.py
│   │
│   ├── decision_engine.py
│   │
│   ├── gpt_agent.py
│   │
│   ├── mission_planner.py
│   │
│   └── app.py
│
├── models/
│   └── trained_models/
│
├── requirements.txt
│
=======
## 🛠️ Technologies Used

### Programming
- **Python 3.10+** - Core development
- **JavaScript** - Frontend interactivity

### AI & Machine Learning
- **PyTorch 2.0+** - Deep learning framework
- **TorchVision** - Computer vision models
- **ResNet18** - Transfer learning backbone
- **GPT-5.6** - Natural language reasoning and reporting
- **scikit-learn** - Data processing and evaluation

### Web Development
- **Flask** - Web server
- **HTML5 + CSS3** - User interface
- **Vanilla JS** - Frontend interactivity

### Data
- **NASA HiRISE Dataset** - 73,031 Mars orbital images
- **8 Terrain Classes** - crater, rocky, sandy, bright dune, dark dune, slope streak, spider, swiss cheese

### Visualization
- **Matplotlib** - Training visualizations
- **Seaborn** - Confusion matrix and statistical plots

---

## 📂 Project Structure

```
AstroPilot-AI/
│
├── data/
│   └── HiRISE/
│       ├── map-proj-v3/              # 73,031 images
│       ├── labels-map-proj-v3.txt
│       └── landmarks_map-proj-v3_classmap.csv
│
├── src/
│   ├── data_loader_hirise.py         # Data loading & preprocessing
│   ├── vision_model.py               # ResNet18 model architecture
│   ├── decision_engine.py            # Risk assessment & decision logic
│   ├── gpt_agent.py                  # GPT-5.6 mission reporting
│   ├── mission_planner.py            # Mission scenario generation
│   └── utils.py                      # Helper functions
│
├── web_app/
│   ├── app.py                        # Flask web application
│   ├── templates/
│   │   └── index.html                # Web interface
│   └── uploads/                      # Temporary uploads
│
├── models/
│   └── trained_models/
│       └── hirise_full_best_model.pth  # Trained model (87.77% accuracy)
│
├── train_hirise.py                   # Full training script
├── train_hirise_fast.py              # Fast training for testing
├── test_evaluation.py                # Model evaluation & visualization
├── predict_hirise.py                 # Single image prediction (CLI)
├── run.py                            # Quick run menu interface
├── create_mock_data.py               # Generate test data
├── requirements.txt
├── .gitignore
>>>>>>> 22ab4f3ab5eab8abd76eee86282351aa1f68dd11
└── README.md
```

---

<<<<<<< HEAD
# 🧩 Code Components

## 1. data_loader.py

**Location:**

```
src/data_loader.py
```

### Purpose:

Loads and prepares NASA HiRISE Mars imagery.

Functions:

* Read images.
* Load labels.
* Prepare data for AI processing.

---

## 2. vision_model.py

**Location:**

```
src/vision_model.py
```

### Purpose:

Computer Vision module.

Responsibilities:

* Analyze planetary images.
* Extract terrain information.
* Identify environmental features.

Input:

```
Mars Image
```

Output:

```
Terrain:
Rocky Area

Risk:
High
=======
## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **8GB+ RAM** (16GB recommended)
- **10GB+ free disk space** (for dataset)
- **Optional:** NVIDIA GPU for faster training

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/AstroPilot-AI.git
cd AstroPilot-AI

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download HiRISE dataset (optional - for training)
# Download from: https://zenodo.org/record/2538136
# Extract to: data/HiRISE/
```

### Web Interface (Recommended)

```bash
# 1. Run the web application
cd web_app
python app.py

# 2. Open your browser and go to:
http://localhost:5000

# 3. Upload an image, adjust parameters, and click "Analyze"
```

### Command Line Interface

```bash
# Quick menu interface
python run.py

# Predict terrain from image
python predict_hirise.py path/to/image.jpg

# With custom parameters
python predict_hirise.py path/to/image.jpg --battery 80 --comm-delay 15

# Evaluate model on test set
python test_evaluation.py
```

### Training

```bash
# Fast training (2000 samples, 3 epochs) - ~10 minutes
python train_hirise_fast.py

# Full training (73,031 samples, 50 epochs) - ~20-30 hours on CPU
python train_hirise.py
>>>>>>> 22ab4f3ab5eab8abd76eee86282351aa1f68dd11
```

---

<<<<<<< HEAD
## 3. decision_engine.py

**Location:**

```
src/decision_engine.py
```

### Purpose:

Autonomous mission decision system.

Combines:

* Terrain analysis
* Mission parameters
* Safety constraints

Output:

```
Decision:
Change Route
```

---

## 4. gpt_agent.py

**Location:**

```
src/gpt_agent.py
```

### Purpose:

AI Mission Officer powered by GPT-5.6.

Responsibilities:

* Explain decisions.
* Generate mission reports.
* Provide recommendations.

---

## 5. mission_planner.py

**Location:**

```
src/mission_planner.py
```

### Purpose:

Creates mission scenarios for:

* Mars Rover
* Satellite Observation
* Lander Operations

Example:

```
Mission:
Mars Exploration

Battery:
40%

Communication Delay:
15 minutes
```

---

## 6. app.py

**Location:**

```
src/app.py
```

### Purpose:

Main application interface.

Connects:

* User Interface
* AI Models
* Mission Engine
* GPT Agent

---

# 🛠️ Technologies Used

## Programming

* Python
* JavaScript

## AI

* GPT-5.6
* Computer Vision
* Machine Learning

## Data

* NASA HiRISE Mars Orbital Dataset

## Backend

* FastAPI

## Frontend

* React / Firebase Hosting

---

# 🚀 Development Roadmap

## Phase 1 — Data Preparation

✅ Download NASA HiRISE Dataset
⬜ Build data loader
⬜ Visualize images and labels

## Phase 2 — AI Vision

⬜ Terrain classification
⬜ Feature extraction

## Phase 3 — Autonomous Decisions

⬜ Risk scoring
⬜ Route recommendations

## Phase 4 — GPT Mission Assistant

⬜ Mission explanation
⬜ AI-generated reports

## Phase 5 — Web Deployment

⬜ Firebase frontend
⬜ Backend deployment
⬜ Public demo


=======
## 📊 Results

### Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **87.77%** |
| Test Loss | 0.5171 |
| Training Time (Fast) | ~10 minutes |
| Training Time (Full) | ~20-30 hours (CPU) |
| Dataset Size | 73,031 images |
| Number of Classes | 8 terrain types |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| bright dune | 100.00% | 33.43% | 50.11% | 350 |
| crater | 83.53% | 28.47% | 42.47% | 980 |
| dark dune | 93.24% | 30.26% | 45.70% | 228 |
| impact ejecta | 100.00% | 4.35% | 8.33% | 46 |
| slope streak | 85.19% | 24.68% | 38.27% | 466 |
| spider | 92.00% | 24.21% | 38.33% | 95 |
| swiss cheese | 98.78% | 35.22% | 51.92% | 230 |
| unknown | 87.69% | 99.36% | 93.16% | 12,212 |

### Training History

The model shows steady improvement over epochs:
- **Start Accuracy (Epoch 1):** ~20%
- **Best Validation Accuracy:** 87.77%
- **Early Stopping:** After 11 epochs (no improvement)

---

## 🔧 How Codex & GPT-5.6 Were Used

### Codex Integration

**Codex** was used throughout the development process to:
- **Accelerate Development:** Generated boilerplate code and common patterns
- **Debugging:** Helped identify and fix issues in model training
- **Documentation:** Assisted in writing comprehensive docstrings and README
- **Web Development:** Generated Flask templates and frontend code

### GPT-5.6 Integration

**GPT-5.6** powers the mission assistant through:

1. **Mission Report Generation**
   - Converts raw risk assessments into human-readable reports
   - Provides natural language explanations for autonomous decisions

2. **Decision Explanation**
   - Explains why certain decisions were made
   - Provides actionable recommendations to mission operators

3. **Enhanced Analysis**
   - Generates detailed terrain descriptions
   - Creates operational recommendations for different terrain types

**/feedback Session ID:** `[INSERT_YOUR_SESSION_ID_HERE]`

---

## 🎯 Use Cases

- **Mars Rover Missions:** Assist in navigation and hazard avoidance
- **Satellite Operations:** Analyze terrain from orbital imagery
- **Lander Missions:** Evaluate landing site safety
- **Education:** AI in space exploration courses
- **Research:** Planetary science and terrain classification

---

## 🌟 Future Improvements

- Real satellite telemetry integration
- Real rover navigation simulation
- Reinforcement learning for autonomous exploration
- Earth observation missions
- Lunar and Mars landing optimization
- Multi-agent coordination (multiple rovers/landers)
- Real-time mission monitoring
- Integration with actual space agency data feeds

---

## 🎥 Demo Video

[Watch the Demo Video](https://youtu.be/YOUR_VIDEO_ID)

---

## 📸 Screenshots

### Web Interface - Home
![Home Page](screenshots/home.png)

### Web Interface - Results
![Results Page](screenshots/results.png)

### AI Mission Report
![Mission Report](screenshots/report.png)

### Training Results
![Training Results](screenshots/training.png)

### Confusion Matrix
![Confusion Matrix](screenshots/confusion_matrix.png)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **NASA** for the HiRISE dataset
- **OpenAI** for Codex and GPT-5.6
- **PyTorch team** for the deep learning framework
- **Devpost** for hosting the OpenAI Build Week challenge

---

## 📧 Contact

- **Developer:** [Your Name]
- **Email:** [your.email@example.com]
- **GitHub:** [https://github.com/yourusername](https://github.com/yourusername)
- **Devpost:** [https://devpost.com/yourusername](https://devpost.com/yourusername)

---

## 📚 Additional Resources

- [OpenAI Build Week](https://openai.devpost.com)
- [NASA HiRISE Dataset](https://zenodo.org/record/2538136)
- [PyTorch Documentation](https://pytorch.org/docs)
- [Flask Documentation](https://flask.palletsprojects.com)

---

**Built with ❤️ for the OpenAI Build Week Challenge using Codex & GPT-5.6**
```


```
# Core ML & Deep Learning
torch>=2.0.0
torchvision>=0.15.0

# Web Framework
flask>=2.2.0
werkzeug>=2.2.0

# Data Processing
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0

# Image Processing
Pillow>=9.5.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
tqdm>=4.64.0
requests>=2.28.0
python-dotenv>=0.21.0

# OpenAI (for GPT-5.6)
openai>=1.0.0

# Development (optional)
pytest>=7.2.0
black>=23.0.0
flake8>=6.0.0
>>>>>>> 22ab4f3ab5eab8abd76eee86282351aa1f68dd11
