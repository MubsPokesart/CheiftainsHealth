import os
from pathlib import Path

class Config:
    """Application configuration."""
    BASE_DIR = Path(__file__).resolve().parent
    MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
    STATIC_DIR = os.path.join(BASE_DIR, "static")
    TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
    TRAINING_CSV_PATH = os.path.join(BASE_DIR, "db", "dataset_triage.csv")
    
    # Model configurations
    DEFAULT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
    NUM_LABELS = 2
    ID2LABEL = {0: 'non-urgent', 1: 'urgent'}
    LABEL2ID = {'non-urgent': 0, 'urgent': 1}
    MAX_LENGTH = 512  # Maximum sequence length
    
    # Training hyperparameters
    BATCH_SIZE = 16  # Training batch size
    EVAL_BATCH_SIZE = 32  # Evaluation batch size
    TRAIN_EPOCHS = 3  # Number of training epochs
    LEARNING_RATE = 2e-5  # Learning rate
    WARMUP_STEPS = 500  # Number of warmup steps
    WEIGHT_DECAY = 0.01  # Weight decay for regularization
    GRADIENT_ACCUMULATION_STEPS = 1  # Number of steps to accumulate gradients
    MAX_GRAD_NORM = 1.0  # Maximum gradient norm for clipping
    
    # Flask configurations
    SECRET_KEY = 'your-secret-key-here'  # Change in production
    DEBUG = True