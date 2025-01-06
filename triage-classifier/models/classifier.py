import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import Config

class TriageClassifier:
    """Main classifier for triage prediction."""
    
    def __init__(self):
        self.config = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Try to load the tokenizer and model from the default Hugging Face model
            print("Loading model from default HuggingFace model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.DEFAULT_MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.DEFAULT_MODEL_NAME,
                num_labels=self.config.NUM_LABELS
            ).to(self.device)
            
        except Exception as e:
            print(f"Error loading default model: {str(e)}")
            raise RuntimeError("Failed to load the model. Please ensure the model files are available.")
    
    def predict(self, text):
        """Predict triage urgency for given text."""
        try:
            # Tokenize and prepare input
            encodings = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**encodings)
                predictions = torch.argmax(outputs.logits, dim=-1)
            
            return {
                "label": self.config.ID2LABEL[predictions.item()],
                "confidence": torch.softmax(outputs.logits, dim=-1).max().item()
            }
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")