"""
Data loading and preprocessing utilities for emotion classification
"""

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import DATASET_NAME, LABEL_MAP, MODEL_NAME, PLOTS_DIR


class EmotionDataLoader:
    def __init__(self):
        self.label_map = LABEL_MAP
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Create plots directory if it doesn't exist
        os.makedirs(PLOTS_DIR, exist_ok=True)
    
    def load_data(self):
        """Load emotion dataset from Hugging Face"""
        print("Loading emotion dataset...")
        
        splits = {
            'train': 'split/train-00000-of-00001.parquet',
            'validation': 'split/validation-00000-of-00001.parquet',
            'test': 'split/test-00000-of-00001.parquet'
        }
        
        train_df = pd.read_parquet(f"hf://datasets/{DATASET_NAME}/" + splits["train"])
        validation_df = pd.read_parquet(f"hf://datasets/{DATASET_NAME}/" + splits["validation"])
        test_df = pd.read_parquet(f"hf://datasets/{DATASET_NAME}/" + splits["test"])
        
        # Add label text mapping
        for df in [train_df, validation_df, test_df]:
            df['label_text'] = df['label'].map(self.label_map)
        
        print(f"Train set size: {len(train_df)}")
        print(f"Validation set size: {len(validation_df)}")
        print(f"Test set size: {len(test_df)}")
        
        return train_df, validation_df, test_df
    
    def explore_data(self, train_df, validation_df, test_df):
        """Perform exploratory data analysis"""
        print("\n=== Data Exploration ===")
        
        # Display basic information
        print("\nTrain DataFrame Head:")
        print(train_df.head())
        
        print(f"\nDataset shapes:")
        print(f"Train: {train_df.shape}")
        print(f"Validation: {validation_df.shape}")
        print(f"Test: {test_df.shape}")
        
        # Check for missing values
        print(f"\nMissing values:")
        print(f"Train: {train_df.isnull().sum().sum()}")
        print(f"Validation: {validation_df.isnull().sum().sum()}")
        print(f"Test: {test_df.isnull().sum().sum()}")
        
        # Analyze text lengths
        self._analyze_text_lengths(train_df, validation_df, test_df)
        
        # Plot label distributions
        self._plot_label_distribution(train_df, validation_df, test_df)
    
    def _analyze_text_lengths(self, train_df, validation_df, test_df):
        """Analyze text length statistics"""
        print("=== Text Length Analysis ===")
        
        for name, df in [("Train", train_df), ("Validation", validation_df), ("Test", test_df)]:
            df['text_length'] = df['text'].apply(lambda x: len(x.split()))
            print(f"{name} text length statistics:")
            print(df['text_length'].describe())
            
            # Plot text length distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(df['text_length'], bins=50, kde=True)
            plt.title(f'Distribution of Text Lengths in {name} Data')
            plt.xlabel('Text Length (number of words)')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(f'{PLOTS_DIR}/text_length_distribution_{name.lower()}.png')
            plt.close()
        
        # Plot text length vs label relationship
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='label_text', y='text_length', data=train_df)
        plt.title('Text Length vs. Label in Training Data')
        plt.xlabel('Emotion Label')
        plt.ylabel('Text Length (words)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/text_length_vs_label.png')
        plt.close()
    
    def _plot_label_distribution(self, train_df, validation_df, test_df):
        """Plot label distribution across datasets"""
        # Add dataset identifier
        train_df_plot = train_df.copy()
        validation_df_plot = validation_df.copy()
        test_df_plot = test_df.copy()
        
        train_df_plot['dataset'] = 'Train'
        validation_df_plot['dataset'] = 'Validation'
        test_df_plot['dataset'] = 'Test'
        
        combined_df = pd.concat([train_df_plot, validation_df_plot, test_df_plot])
        
        plt.figure(figsize=(12, 8))
        sns.countplot(data=combined_df, x='label_text', hue='dataset', 
                     order=self.label_map.values())
        plt.title('Distribution of Labels Across All Datasets')
        plt.xlabel('Emotion Label')
        plt.ylabel('Count')
        plt.legend(title='Dataset')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/label_distribution.png')
        plt.close()
        
        print(f"\nPlots saved to {PLOTS_DIR}/ directory")
    
    def tokenize_function(self, examples):
        """Tokenize text data"""
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)
    
    def prepare_datasets(self, train_df, validation_df, test_df):
        """Convert pandas DataFrames to Hugging Face datasets and tokenize"""
        print("Preparing and tokenizing datasets...")
        
        emotion_dataset = DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'validation': Dataset.from_pandas(validation_df),
            'test': Dataset.from_pandas(test_df)
        })
        
        tokenized_datasets = emotion_dataset.map(self.tokenize_function, batched=True)
        
        print("Datasets tokenized successfully!")
        return tokenized_datasets
