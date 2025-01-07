import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import Config
from utils.preprocessor import DataPreprocessor
from models.trainer import TriageTrainer

def train_model():
    """Train and save a new model using the configured CSV path."""
    config = Config()
    
    # Verify CSV file exists
    if not os.path.exists(config.TRAINING_CSV_PATH):
        raise FileNotFoundError(
            f"Training data not found at {config.TRAINING_CSV_PATH}. "
            "Please ensure the dataset_triage.csv file is in the db directory."
        )
    
    print(f"Using training data from: {config.TRAINING_CSV_PATH}")
    
    print("Loading tokenizer and base model...")
    # Updated model loading configuration
    model_kwargs = {
        "trust_remote_code": True,
        "force_download": False
    }
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.DEFAULT_MODEL_NAME, 
        **model_kwargs
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config.DEFAULT_MODEL_NAME,
        num_labels=config.NUM_LABELS,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
        **model_kwargs
    )
    
    # Set up data preprocessor
    print("Preparing data...")
    preprocessor = DataPreprocessor(tokenizer)
    train_dataset, eval_dataset = preprocessor.prepare_data(config.TRAINING_CSV_PATH)
    
    # Calculate class weights using numpy
    labels = train_dataset.labels
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    
    # Calculate weights: 1 / frequency
    class_weights = torch.FloatTensor(total_samples / (len(unique_labels) * counts))
    print("\nClass weights:")
    for label_id, weight in enumerate(class_weights):
        label_name = config.ID2LABEL[label_id]
        print(f"{label_name}: {weight:.4f}")
    
    # Set up trainer
    print("\nSetting up trainer...")
    trainer = TriageTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        class_weights=class_weights
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # Train the model
    print("Starting training...")
    trainer.train(config.MODEL_DIR)
    
    # Save the model and tokenizer
    print(f"Saving model to {config.MODEL_DIR}...")
    model.save_pretrained(config.MODEL_DIR, push_to_hub=False)
    tokenizer.save_pretrained(config.MODEL_DIR, push_to_hub=False)
    
    print("Training complete!")
    return model, tokenizer

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        import traceback
        traceback.print_exc()