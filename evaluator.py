"""
Model evaluation utilities for emotion classification
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
from config import LABEL_MAP, PLOTS_DIR


class EmotionEvaluator:
    def __init__(self):
        self.label_map = LABEL_MAP
        os.makedirs(PLOTS_DIR, exist_ok=True)
    
    def get_predictions(self, trainer, tokenized_test_dataset):
        """Get model predictions on test set"""
        print("Getting predictions on test set...")
        
        test_predictions = trainer.predict(tokenized_test_dataset)
        predictions = np.argmax(test_predictions.predictions, axis=-1)
        true_labels = tokenized_test_dataset["label"]
        
        return predictions, true_labels
    
    def print_classification_report(self, true_labels, predictions):
        """Print detailed classification report"""
        print("\n" + "="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)
        
        report = classification_report(
            true_labels, 
            predictions, 
            target_names=list(self.label_map.values()),
            digits=4
        )
        print(report)
        
        return report
    
    def plot_confusion_matrix(self, true_labels, predictions, save_path=None):
        """Plot and save confusion matrix"""
        print("Generating confusion matrix...")
        
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=list(self.label_map.values()),
            yticklabels=list(self.label_map.values())
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix - Emotion Classification')
        plt.tight_layout()
        
        if save_path is None:
            save_path = f'{PLOTS_DIR}/confusion_matrix.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {save_path}")
        return cm
    
    def calculate_per_class_metrics(self, true_labels, predictions):
        """Calculate detailed per-class metrics"""
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        
        # Overall accuracy
        accuracy = accuracy_score(true_labels, predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None
        )
        
        # Create detailed metrics dictionary
        metrics = {
            'overall_accuracy': accuracy,
            'per_class_metrics': {}
        }
        
        print("\n" + "="*60)
        print("DETAILED PER-CLASS METRICS")
        print("="*60)
        print(f"{'Emotion':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 60)
        
        for i, emotion in self.label_map.items():
            metrics['per_class_metrics'][emotion] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': support[i]
            }
            
            print(f"{emotion:<12} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<10}")
        
        print("-" * 60)
        print(f"{'Overall':<12} {'':<10} {'':<10} {'':<10} {sum(support):<10}")
        print(f"{'Accuracy':<12} {'':<10} {'':<10} {accuracy:<10.4f}")
        
        return metrics
    
    def evaluate_model(self, trainer, tokenized_test_dataset):
        """Complete model evaluation pipeline"""
        print("\n🔍 Starting comprehensive model evaluation...")
        
        # Get predictions
        predictions, true_labels = self.get_predictions(trainer, tokenized_test_dataset)
        
        # Print classification report
        classification_rep = self.print_classification_report(true_labels, predictions)
        
        # Plot confusion matrix
        confusion_mat = self.plot_confusion_matrix(true_labels, predictions)
        
        # Calculate detailed metrics
        detailed_metrics = self.calculate_per_class_metrics(true_labels, predictions)
        
        print("\n✅ Model evaluation completed!")
        
        return {
            'predictions': predictions,
            'true_labels': true_labels,
            'classification_report': classification_rep,
            'confusion_matrix': confusion_mat,
            'detailed_metrics': detailed_metrics
        }
