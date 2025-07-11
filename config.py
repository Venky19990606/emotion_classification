"""
Configuration file for emotion classification project
"""

# Model configuration
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 6
MAX_LENGTH = 512

# Dataset configuration
DATASET_NAME = "dair-ai/emotion"
LABEL_MAP = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

# Training configuration
TRAINING_CONFIG = {
    "output_dir": "output",
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "num_train_epochs": 2,
    "weight_decay": 0.01,
    "eval_strategy": "steps",
    "save_strategy": "steps",
    "eval_steps": 500,
    "save_steps": 500,
    "load_best_model_at_end": True,
    "report_to": "none",
    "push_to_hub": False,
    "fp16": True
}

# Paths
MODEL_SAVE_PATH = "saved_model"
PLOTS_DIR = "plots"
