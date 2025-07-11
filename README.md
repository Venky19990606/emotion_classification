# Emotion Classification with BERT

A professional implementation of emotion classification using BERT (Bidirectional Encoder Representations from Transformers). This project fine-tunes a pre-trained BERT model to classify text into six different emotions: sadness, joy, love, anger, fear, and surprise.

## Project Overview

This project provides a complete pipeline for:
- Loading and exploring the emotion dataset
- Fine-tuning BERT for emotion classification
- Comprehensive model evaluation
- Easy-to-use inference capabilities
- Professional code structure with modular design

## Dataset

The project uses the [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset from Hugging Face, which contains:
- **Training set**: 16,000 examples
- **Validation set**: 2,000 examples  
- **Test set**: 2,000 examples

### Emotion Labels
- 0: Sadness
- 1: Joy
- 2: Love
- 3: Anger
- 4: Fear
- 5: Surprise

## Quick Start

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended for training)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd emotion_classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

#### Basic Training
```bash
python train.py
```

#### Training Options
```bash
# Skip data exploration
python train.py --no-explore

# Skip evaluation after training
python train.py --no-evaluate

# Load existing model without training
python train.py --skip-training
```

The training process includes:
1. **Data Loading**: Automatically downloads and loads the emotion dataset
2. **Data Exploration**: Generates visualizations and statistics (saved to `plots/`)
3. **Model Training**: Fine-tunes BERT-base-uncased for 2 epochs
4. **Model Evaluation**: Comprehensive evaluation with metrics and visualizations
5. **Model Saving**: Saves the trained model to `saved_model/`

### Making Predictions

#### Single Prediction
```bash
python predict.py --model-path /path/to/your/model --text "I am so happy today!"
```

#### Interactive Mode
```bash
python predict.py --model-path /path/to/your/model --interactive
```

#### Demo Predictions
```bash
python predict.py --model-path /path/to/your/model --demo
```

### Programmatic Usage

```python
from inference import EmotionPredictor

# Initialize predictor
predictor = EmotionPredictor(model_path = '/path/to/your/model')
predictor.load_model()

# Single prediction
predictions = predictor.predict_single("I love this movie!")
print(f"Top emotion: {predictions[0]['emotion']}")
print(f"Confidence: {predictions[0]['confidence']:.4f}")

# Batch predictions
texts = ["I'm sad", "Great news!", "I'm scared"]
results = predictor.predict_batch(texts)
for result in results:
    print(f"Text: {result['text']}")
    print(f"Emotion: {result['top_emotion']}")
```

## 📁 Project Structure

```
emotion_classification/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── config.py                # Configuration settings
├── data_loader.py           # Data loading and preprocessing
├── trainer.py               # Model training utilities
├── evaluator.py             # Model evaluation utilities
├── inference.py             # Inference and prediction utilities
├── train.py                 # Main training script
├── predict.py               # Simple prediction script
├── output/                  # Training outputs and checkpoints
├── saved_model/             # Saved trained model
└── plots/                   # Generated visualizations
    ├── label_distribution.png
    ├── text_length_distribution_*.png
    ├── text_length_vs_label.png
    └── confusion_matrix.png
```

## Configuration

The project uses a centralized configuration system in `config.py`. Key settings include:

- **Model**: BERT-base-uncased
- **Training epochs**: 2
- **Batch size**: 32
- **Learning rate**: 2e-5
- **Evaluation strategy**: Every 500 steps

You can modify these settings by editing the `config.py` file.

## Model Performance

The fine-tuned BERT model achieves strong performance across all emotion categories:

- **Overall Accuracy**: ~92-94%
- **Balanced performance** across all six emotion classes
- **Detailed metrics** including precision, recall, and F1-score for each emotion

## Evaluation Features

The project provides comprehensive evaluation capabilities:

1. **Classification Report**: Detailed per-class metrics
2. **Confusion Matrix**: Visual representation of model performance
3. **Data Visualizations**: 
   - Label distribution across datasets
   - Text length analysis
   - Performance metrics visualization