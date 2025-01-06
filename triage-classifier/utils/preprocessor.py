import pandas as pd
from sklearn.model_selection import train_test_split
from models.dataset import TriageDataset
from config import Config

class DataPreprocessor:
    """Handles data preprocessing for the triage classifier."""
    
    def __init__(self, tokenizer):
        self.config = Config()
        self.tokenizer = tokenizer
    
    def prepare_data(self, data_path, test_size=0.2):
        """Prepare data for training and evaluation."""
        # Read and clean data
        df = pd.read_csv(data_path)
        df['question'] = df['question'].astype('string')
        df['triage'] = df['triage'].map(
            lambda x: self.config.LABEL2ID[x] if x in self.config.LABEL2ID else x
        )
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df["question"],
            df["triage"],
            test_size=test_size,
            random_state=42
        )
        
        # Tokenize
        train_encodings = self.tokenizer(
            list(train_texts),
            truncation=True,
            padding=True,
            max_length=self.config.MAX_LENGTH
        )
        val_encodings = self.tokenizer(
            list(val_texts),
            truncation=True,
            padding=True,
            max_length=self.config.MAX_LENGTH
        )
        
        # Create datasets
        train_dataset = TriageDataset(train_encodings, train_labels.values)
        val_dataset = TriageDataset(val_encodings, val_labels.values)
        
        return train_dataset, val_dataset