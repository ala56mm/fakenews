"""
Improved Model Training Module for Fake News Detection System
Uses TF-IDF with proper train/test split and multiple classifiers
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)


class TextPreprocessor:
    """Handles text preprocessing operations"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """Apply all preprocessing steps"""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        words = text.split()
        words = [w for w in words if w not in self.stop_words and len(w) > 2]
        words = [self.lemmatizer.lemmatize(w) for w in words]
        
        return ' '.join(words)


class FakeNewsTrainer:
    """Handles model training with multiple classifiers"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                C=1.0
            ),
            'Multinomial Naive Bayes': MultinomialNB(alpha=0.1)
        }
        self.best_model = None
        self.best_model_name = None
    
    def load_data(self, filepath):
        """Load dataset from CSV"""
        print(f"\n[1] Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"    Total samples: {len(df)}")
        print(f"    Label distribution:")
        for label, count in df['label'].value_counts().items():
            print(f"      {label}: {count} ({count/len(df)*100:.1f}%)")
        return df
    
    def preprocess_data(self, df):
        """Preprocess all text"""
        print("\n[2] Preprocessing text...")
        df['processed_text'] = df['text'].apply(self.preprocessor.clean_text)
        df = df[df['processed_text'].str.strip() != '']
        print(f"    Samples after preprocessing: {len(df)}")
        return df
    
    def train_models(self, df, test_size=0.2, random_state=42):
        """
        Train multiple models and compare performance
        """
        print("\n[3] Splitting data (80% train / 20% test)...")
        
        X = df['processed_text']
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"    Training samples: {len(X_train)}")
        print(f"    Testing samples: {len(X_test)}")
        print(f"    Train label distribution: FAKE={sum(y_train=='FAKE')}, REAL={sum(y_train=='REAL')}")
        print(f"    Test label distribution: FAKE={sum(y_test=='FAKE')}, REAL={sum(y_test=='REAL')}")
        
        print("\n[4] Vectorizing text with TF-IDF...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        print(f"    Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print(f"    Feature matrix shape: {X_train_tfidf.shape}")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n[5] Training {name}...")
            model.fit(X_train_tfidf, y_train)
            
            y_pred = model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            
            cross_val = cross_val_score(model, X_train_tfidf, y_train, cv=3)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cross_val.mean(),
                'cv_std': cross_val.std()
            }
            
            print(f"\n    {'='*50}")
            print(f"    {name.upper()} RESULTS")
            print(f"    {'='*50}")
            print(f"    Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"    Cross-Val Accuracy: {cross_val.mean():.4f} (+/- {cross_val.std()*2:.4f})")
            
            print(f"\n    Classification Report:")
            print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))
            
            print(f"    Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
            print(f"                    Predicted")
            print(f"                 FAKE  REAL")
            print(f"    Actual FAKE   {cm[0][0]:3d}  {cm[0][1]:3d}")
            print(f"    Actual REAL   {cm[1][0]:3d}  {cm[1][1]:3d}")
        
        self.best_model_name = max(results, key=lambda k: results[k]['accuracy'])
        self.best_model = results[self.best_model_name]['model']
        
        print(f"\n{'='*60}")
        print(f"    BEST MODEL: {self.best_model_name}")
        print(f"    Best Accuracy: {results[self.best_model_name]['accuracy']*100:.2f}%")
        print(f"{'='*60}")
        
        return results, X_test, y_test
    
    def save_model(self, model_path='model/model.pkl', vectorizer_path='model/vectorizer.pkl'):
        """Save trained model and vectorizer"""
        print(f"\n[6] Saving model to {model_path}...")
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.best_model,
                'model_name': self.best_model_name,
                'vectorizer': self.vectorizer
            }, f)
        
        print(f"    Saved: {model_path}")
        print(f"    Model: {self.best_model_name}")
    
    def run_training(self, data_path='data/news.csv'):
        """Run complete training pipeline"""
        print("\n" + "="*60)
        print("FAKE NEWS DETECTION - IMPROVED MODEL TRAINING")
        print("="*60)
        
        df = self.load_data(data_path)
        df = self.preprocess_data(df)
        results, X_test, y_test = self.train_models(df)
        self.save_model()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        
        return results


def main():
    """Main training function"""
    trainer = FakeNewsTrainer()
    results = trainer.run_training()


if __name__ == "__main__":
    main()
