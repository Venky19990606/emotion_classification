"""
Inference utilities for emotion classification
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from config import LABEL_MAP, MODEL_SAVE_PATH


class EmotionPredictor:
    def __init__(self, model_path=None):
        self.model_path = model_path or MODEL_SAVE_PATH
        self.label_map = LABEL_MAP
        self.model = None
        self.tokenizer = None
        self.classifier = None
        
    def load_model(self):
        """Load the trained model and tokenizer"""
        print(f"Loading model from {self.model_path}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            
            # Create pipeline for easy inference
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                return_all_scores=True
            )
            
            print("Model loaded successfully! ✅")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please make sure the model is trained and saved properly.")
            raise
    
    def predict_single(self, text):
        """Predict emotion for a single text"""
        if self.classifier is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Get prediction
        results = self.classifier(text)
        
        # Parse results
        predictions = []
        for result in results[0]:  # results[0] because return_all_scores=True returns list of lists
            label_id = int(result['label'].split('_')[-1])  # Extract number from LABEL_X
            emotion = self.label_map[label_id]
            confidence = result['score']
            predictions.append({
                'emotion': emotion,
                'confidence': confidence
            })
        
        # Sort by confidence (highest first)
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return predictions
    
    def predict_batch(self, texts):
        """Predict emotions for multiple texts"""
        if self.classifier is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        results = []
        for text in texts:
            prediction = self.predict_single(text)
            results.append({
                'text': text,
                'predictions': prediction,
                'top_emotion': prediction[0]['emotion'],
                'confidence': prediction[0]['confidence']
            })
        
        return results
    
    def predict_with_details(self, text, top_k=3):
        """Predict with detailed output showing top-k emotions"""
        predictions = self.predict_single(text)
        
        print(f"\nText: \"{text}\"")
        print("-" * 50)
        print(f"Top {min(top_k, len(predictions))} predicted emotions:")
        
        for i, pred in enumerate(predictions[:top_k]):
            print(f"{i+1}. {pred['emotion'].capitalize()}: {pred['confidence']:.4f}")
        
        return predictions
    
    def demo_predictions(self):
        """Run demo predictions on example sentences"""
        example_sentences = [
            "I am feeling very sad today.",
            "This is the best day of my life, I am so happy!",
            "I am so in love with this movie.",
            "That made me so angry, I could scream.",
            "I am really scared about the presentation tomorrow.",
            "Wow, that was an unexpected surprise!"
        ]
        
        print("\n" + "="*60)
        print("DEMO PREDICTIONS")
        print("="*60)
        
        results = []
        for sentence in example_sentences:
            predictions = self.predict_with_details(sentence, top_k=2)
            results.append({
                'text': sentence,
                'predictions': predictions
            })
            print()
        
        return results
    
    def interactive_prediction(self):
        """Interactive prediction mode"""
        print("\n" + "="*60)
        print("INTERACTIVE EMOTION PREDICTION")
        print("="*60)
        print("Enter text to predict emotions (type 'quit' to exit)")
        print("-" * 60)
        
        while True:
            text = input("\nEnter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! ")
                break
            
            if not text:
                print("Please enter some text.")
                continue
            
            try:
                self.predict_with_details(text)
            except Exception as e:
                print(f"Error during prediction: {e}")


def main():
    """Main function for standalone inference"""
    predictor = EmotionPredictor()
    
    try:
        predictor.load_model()
        predictor.demo_predictions()
        predictor.interactive_prediction()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have trained and saved a model first by running train.py")


if __name__ == "__main__":
    main()
