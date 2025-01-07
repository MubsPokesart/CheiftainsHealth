from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import evaluate
import numpy as np
import torch
from config import Config

class WeightedTrainer(Trainer):
    """Custom trainer that implements weighted loss calculation."""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override compute_loss to implement class weights with updated signature."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None and torch.is_tensor(self.class_weights):
            # Apply class weights to the loss
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device),
                reduction='mean'
            )
        else:
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        
        # If num_items_in_batch is provided, scale the loss accordingly
        if num_items_in_batch is not None and num_items_in_batch > 0:
            loss = loss * logits.size(0) / num_items_in_batch
            
        return (loss, outputs) if return_outputs else loss

class TriageTrainer:
    """Trainer class for fine-tuning the triage model."""
    
    def __init__(self, model, train_dataset, eval_dataset=None, class_weights=None):
        self.config = Config()
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.metric = evaluate.load("accuracy")
        self.class_weights = class_weights
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        # Calculate multiple metrics
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = self.metric.compute(
            predictions=predictions, 
            references=labels
        )['accuracy']
        
        # Per-class metrics
        for i, class_name in self.config.ID2LABEL.items():
            class_pred = (predictions == i)
            class_true = (labels == i)
            
            # True positives, false positives, etc.
            tp = np.sum((class_pred) & (class_true))
            fp = np.sum((class_pred) & (~class_true))
            tn = np.sum((~class_pred) & (~class_true))
            fn = np.sum((~class_pred) & (class_true))
            
            # Precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[f'{class_name}_precision'] = precision
            metrics[f'{class_name}_recall'] = recall
            metrics[f'{class_name}_f1'] = f1
        
        return metrics
    
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
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            gradient_checkpointing=True,
            # Disable hub integration
            push_to_hub=False,
            # Add logging
            logging_dir=f"{output_dir}/logs",
            logging_strategy="steps",
            logging_steps=100,
            # Add mixed precision training if GPU is available
            fp16=torch.cuda.is_available(),
            # Add report to hp search
            report_to="none"
        )
        
        # Create trainer with early stopping callback
        trainer = WeightedTrainer(
            class_weights=self.class_weights,
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.01
                )
            ]
        )
        
        return trainer.train()