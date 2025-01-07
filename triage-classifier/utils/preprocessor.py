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
        try:
            # Read and validate data
            df = pd.read_csv(data_path)
            
            # Verify required columns exist
            required_columns = {'question', 'triage'}
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Clean and preprocess
            df['question'] = df['question'].astype('string').str.strip()
            df['triage'] = df['triage'].str.lower().str.strip()
            
            # Validate triage labels
            valid_labels = set(self.config.LABEL2ID.keys())
            invalid_labels = set(df['triage'].unique()) - valid_labels
            if invalid_labels:
                raise ValueError(
                    f"Invalid triage labels found: {invalid_labels}. "
                    f"Valid labels are: {valid_labels}"
                )
            
            # Convert labels to IDs
            df['triage'] = df['triage'].map(self.config.LABEL2ID)
            
            # Remove rows with missing values
            initial_rows = len(df)
            df = df.dropna()
            if len(df) < initial_rows:
                print(f"Removed {initial_rows - len(df)} rows with missing values")
            
            # Split data
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                df["question"],
                df["triage"],
                test_size=test_size,
                random_state=42,
                stratify=df["triage"]  # Ensure balanced split
            )
            
            # Print dataset statistics
            print(f"\nDataset Statistics:")
            print(f"Total examples: {len(df)}")
            print(f"Training examples: {len(train_texts)}")
            print(f"Validation examples: {len(val_texts)}")
            print("\nLabel distribution:")
            for label, id in self.config.LABEL2ID.items():
                count = len(df[df['triage'] == id])
                percentage = (count / len(df)) * 100
                print(f"{label}: {count} ({percentage:.1f}%)")
            
            # Tokenize
            train_encodings = self.tokenizer(
                list(train_texts),
                truncation=True,
                padding=True
            )
            val_encodings = self.tokenizer(
                list(val_texts),
                truncation=True,
                padding=True
            )
            
            # Create datasets
            train_dataset = TriageDataset(train_encodings, train_labels.values)
            val_dataset = TriageDataset(val_encodings, val_labels.values)
            
            return train_dataset, val_dataset
            
        except Exception as e:
            print(f"Error preprocessing data: {str(e)}")
            raise