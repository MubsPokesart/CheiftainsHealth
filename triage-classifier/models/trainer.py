from transformers import Trainer, TrainingArguments
import evaluate
import numpy as np
from config import Config

class TriageTrainer:
    """Trainer class for fine-tuning the triage model."""
    
    def __init__(self, model, train_dataset, eval_dataset=None):
        self.config = Config()
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.metric = evaluate.load("accuracy")
        
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)
    
    def train(self, output_dir):
        """Train the model."""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.TRAIN_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.EVAL_BATCH_SIZE,
            warmup_steps=self.config.WARMUP_STEPS,
            weight_decay=self.config.WEIGHT_DECAY,
            learning_rate=self.config.LEARNING_RATE,
            gradient_accumulation_steps=self.config.GRADIENT_ACCUMULATION_STEPS,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            gradient_checkpointing=True
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics
        )
        
        return trainer.train()
