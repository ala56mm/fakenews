# Fake News Detection System

A machine learning-based web application that detects whether news articles are fake or real using TF-IDF vectorization and Logistic Regression.

## Features

- Text preprocessing (lowercase, remove punctuation, stopwords)
- TF-IDF feature extraction with n-grams
- Logistic Regression classifier
- Web interface for real-time predictions
- Confidence score for predictions

## Project Structure

```
fakenews/
├── data/
│   └── news.csv              # Dataset file
├── model/
│   ├── model.pkl            # Trained model (generated)
│   └── vectorizer.pkl       # TF-IDF vectorizer (generated)
├── templates/
│   └── index.html           # Web interface
├── app.py                    # Flask web application
├── train.py                  # Model training script
├── predict.py                # Prediction module
├── preprocess.py             # Text preprocessing
├── requirements.txt         # Python dependencies
└── README.md                 # This file
```

## Installation

1. Clone or navigate to the project directory

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Train the Model

Before using the web application, you need to train the model:

```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Train a Logistic Regression model
- Evaluate accuracy
- Save model and vectorizer to `model/` directory

### Step 2: Run the Web Application

Start the Flask server:

```bash
python app.py
```

Open your browser and go to: `http://127.0.0.1:5000`

### Step 3: Test Predictions

- Enter any news headline or article text
- Click "Analyze News"
- View the prediction result with confidence score

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/predict` | POST | Make prediction on provided text |
| `/health` | GET | Health check endpoint |

### Prediction Request

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your news text here"}'
```

### Response

```json
{
  "success": true,
  "label": "REAL",
  "confidence": 94.52
}
```

## Dataset Format

The dataset should be a CSV file with columns:
- `text`: News article text
- `label`: Either "FAKE" or "REAL"

## Model Details

- **Vectorizer**: TF-IDF with max 5000 features, unigrams and bigrams
- **Classifier**: Logistic Regression (max_iter=1000)
- **Test Split**: 20% of data

## How It Works

1. **Preprocessing**: Text is converted to lowercase, punctuation removed, stopwords filtered, and words lemmatized
2. **Vectorization**: TF-IDF converts text to numerical features
3. **Classification**: Logistic Regression predicts FAKE or REAL
4. **Output**: Returns prediction with confidence percentage

## Requirements

- Python 3.8+
- Flask
- pandas
- scikit-learn
- nltk
- numpy

## License

MIT License
