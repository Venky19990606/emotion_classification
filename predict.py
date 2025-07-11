"""
Simple prediction script for emotion classification
"""

import argparse
from inference import EmotionPredictor


def main():
    parser = argparse.ArgumentParser(description="Predict emotions from text")
    
    parser.add_argument(
        "--text", 
        type=str, 
        help="Text to predict emotion for"
    )
    
    parser.add_argument(
        "--interactive", 
        action="store_true", 
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--demo", 
        action="store_true", 
        help="Run demo predictions"
    )
    
    parser.add_argument(
        "--model-path", 
        type=str, 
        default=None,
        help="Path to saved model (default: saved_model/)"
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = EmotionPredictor(model_path=args.model_path)
    
    try:
        predictor.load_model()
        
        if args.text:
            # Single prediction
            predictor.predict_with_details(args.text)
            
        elif args.demo:
            # Demo predictions
            predictor.demo_predictions()
            
        elif args.interactive:
            # Interactive mode
            predictor.interactive_prediction()
            
        else:
            print("No specific mode selected. Running demo predictions...")
            predictor.demo_predictions()
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Trained a model by running: python train.py")
        print("2. The model is saved in the correct location")


if __name__ == "__main__":
    main()
