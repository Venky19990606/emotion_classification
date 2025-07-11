"""
Main training script for emotion classification model
"""

import argparse
import os
from data_loader import EmotionDataLoader
from trainer import EmotionTrainer
from evaluator import EmotionEvaluator


def main(args):
    """Main training pipeline"""
    print("Starting Emotion Classification Training Pipeline")
    print("=" * 60)
    
    # Initialize components
    data_loader = EmotionDataLoader()
    trainer = EmotionTrainer()
    evaluator = EmotionEvaluator()
    
    # Step 1: Load and explore data
    print("Step 1: Loading and exploring data...")
    train_df, validation_df, test_df = data_loader.load_data()
    
    if args.explore_data:
        data_loader.explore_data(train_df, validation_df, test_df)
    
    # Step 2: Prepare datasets
    print("Step 2: Preparing datasets...")
    tokenized_datasets = data_loader.prepare_datasets(train_df, validation_df, test_df)
    
    # Step 3: Load model and setup trainer
    print("\Step 3: Loading model and setting up trainer...")
    trainer.load_model_and_tokenizer()
    trainer.setup_trainer(tokenized_datasets)
    
    # Step 4: Train the model
    if not args.skip_training:
        print("Step 4: Training the model...")
        trainer.train()
        
        # Save the model
        print("Step 5: Saving the model...")
        trainer.save_model()
    else:
        print("Step 4: Skipping training (loading existing model)...")
        trainer.load_saved_model()
    
    # Step 5: Evaluate the model
    if args.evaluate:
        print("Step 6: Evaluating the model...")
        
        # Basic evaluation
        results = trainer.evaluate(tokenized_datasets)
        
        # Comprehensive evaluation
        evaluation_results = evaluator.evaluate_model(trainer.trainer, tokenized_datasets["test"])
        
        print(f"Training pipeline completed!")
        print(f"Final Test Accuracy: {results['eval_accuracy']:.4f}")
    
    print("All done! You can now use the model for inference.")
    print("Run: python inference.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train emotion classification model")
    
    parser.add_argument(
        "--skip-training", 
        action="store_true", 
        help="Skip training and load existing model"
    )
    
    parser.add_argument(
        "--no-explore", 
        dest="explore_data",
        action="store_false", 
        help="Skip data exploration"
    )
    
    parser.add_argument(
        "--no-evaluate", 
        dest="evaluate",
        action="store_false", 
        help="Skip model evaluation"
    )
    
    args = parser.parse_args()
    main(args)
