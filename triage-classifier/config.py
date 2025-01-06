import os
from pathlib import Path

class Config:
    """Application configuration."""
    BASE_DIR = Path(__file__).resolve().parent
    MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
    STATIC_DIR = os.path.join(BASE_DIR, "static")
    TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
    
    # Model configurations
    DEFAULT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
    NUM_LABELS = 2
    ID2LABEL = {0: 'non-urgent', 1: 'urgent'}
    LABEL2ID = {'non-urgent': 0, 'urgent': 1}
    
    # Flask configurations
    SECRET_KEY = 'your-secret-key-here'  # Change in production
    DEBUG = True
