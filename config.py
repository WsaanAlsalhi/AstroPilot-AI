# config.py
import torch

# Paths
DATA_DIR = "data/HiRISE/images/"
LABELS_PATH = "data/HiRISE/labels.csv"
MODEL_SAVE_PATH = "models/trained_models/best_model.pth"

# Training hyperparameters
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 0.001
NUM_WORKERS = 2

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Augmentation settings
IMG_SIZE = 224
RANDOM_HORIZONTAL_FLIP_PROB = 0.5
RANDOM_ROTATION_DEGREES = 10
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2

# Label encoding
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Model
FREEZE_BACKBONE = True