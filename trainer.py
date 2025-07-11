"""
Model training utilities for emotion classification
"""

import os
import numpy as np
import evaluate
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from config import MODEL_NAME, NUM_LABELS, TRAINING_CONFIG, MODEL_SAVE_PATH


class EmotionTrainer:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.num_labels = NUM_LABELS
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.metric = evaluate.load("accuracy")
        
        # Create output directories
        os.makedirs(TRAINING_CONFIG["output_dir"], exist_ok=True)
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    def load_model_and_tokenizer(self):
        """Load pre-trained model and tokenizer"""
        print(f"Loading model and tokenizer: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=self.num_labels
        )
        
        print("Model and tokenizer loaded successfully!")
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)
    
    def setup_trainer(self, tokenized_datasets):
        """Setup the Hugging Face Trainer"""
        print("Setting up trainer...")
        
        training_args = TrainingArguments(**TRAINING_CONFIG)
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )
        
        print("Trainer setup complete!")
    
    def train(self):
        """Train the model"""
        if self.trainer is None:
            raise ValueError("Trainer not setup. Call setup_trainer() first.")
        
        print("Starting the fine-tuning process... ")
        self.trainer.train()
        print("Training completed! ")
    
    def evaluate(self, tokenized_datasets):
        """Evaluate the model on test set"""
        print("Evaluating model on test set...")
        
        results = self.trainer.evaluate(tokenized_datasets["test"])
        test_accuracy = results['eval_accuracy']
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        return results
    
    def save_model(self, save_path=None):
        """Save the trained model and tokenizer"""
        if save_path is None:
            save_path = MODEL_SAVE_PATH
        
        print(f"Saving model to {save_path}...")
        self.trainer.save_model(save_path)
        print(f"Model saved successfully to {save_path}! ")
    
    def load_saved_model(self, model_path=None):
        """Load a previously saved model"""
        if model_path is None:
            model_path = MODEL_SAVE_PATH
        
        print(f"Loading model from {model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        print("Model loaded successfully!")
        return self.model, self.tokenizer
