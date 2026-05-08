"""
Improved Prediction Module for Fake News Detection System
Handles model loading and predictions with confidence scores
"""

import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


class TextPreprocessor:
    """Handles text preprocessing"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """Clean and preprocess text"""
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


class FakeNewsPredictor:
    """Handles loading model and making predictions"""
    
    def __init__(self, model_path='model/model.pkl'):
        self.preprocessor = TextPreprocessor()
        self.model_data = None
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained model from disk"""
        try:
            print(f"Loading model from {model_path}...")
            with open(model_path, 'rb') as f:
                self.model_data = pickle.load(f)
            
            self.model = self.model_data['model']
            self.vectorizer = self.model_data['vectorizer']
            self.model_name = self.model_data.get('model_name', 'Unknown')
            self.preprocessor = TextPreprocessor()
            
            print(f"    Model loaded successfully!")
            print(f"    Model type: {self.model_name}")
            
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model not found at {model_path}. Please run train.py first."
            )
        except Exception as e:
            raise Exception(f"Error loading model: {e}")
    
    def predict(self, text):
        """
        Predict whether text is FAKE or REAL
        
        Args:
            text (str): News article text
            
        Returns:
            dict: Prediction result with label and confidence
        """
        if not text or not text.strip():
            return {
                'label': 'ERROR',
                'confidence': 0.0,
                'error': 'Empty text provided'
            }
        
        processed_text = self.preprocessor.clean_text(text)
        
        if not processed_text:
            return {
                'label': 'ERROR',
                'confidence': 0.0,
                'error': 'Text could not be processed'
            }
        
        text_tfidf = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(text_tfidf)[0]
        
        probabilities = self.model.predict_proba(text_tfidf)[0]
        confidence = max(probabilities) * 100
        
        class_labels = self.model.classes_
        prob_dict = {label: prob * 100 for label, prob in zip(class_labels, probabilities)}
        
        return {
            'label': prediction,
            'confidence': round(confidence, 2),
            'probabilities': {k: round(v, 2) for k, v in prob_dict.items()}
        }
    
    def predict_batch(self, texts):
        """Predict for multiple texts"""
        return [self.predict(text) for text in texts]


def main():
    """Test the prediction module"""
    print("\n" + "="*60)
    print("FAKE NEWS DETECTION - PREDICTION TEST")
    print("="*60)
    
    try:
        predictor = FakeNewsPredictor()
        
        fake_samples = [
            "Doctors shocked to discover incredible weight loss secret",
            "Ancient prophecy predicts world ending next year",
            "Scientists admit this one kitchen ingredient cures all diseases",
            "Government hides evidence of UFO contact from public",
            "Miracle supplement makes you lose 30 pounds in one week"
        ]
        
        real_samples = [
            "Scientists discover new species of butterfly in Amazon rainforest",
            "Stock market closes higher following positive economic data",
            "City council approves new budget for public transportation",
            "Federal Reserve maintains interest rates unchanged",
            "Researchers find link between diet and longevity in new study"
        ]
        
        print("\n" + "-"*60)
        print("TESTING FAKE NEWS SAMPLES:")
        print("-"*60)
        for text in fake_samples:
            result = predictor.predict(text)
            status = "[PASS]" if result['label'] == 'FAKE' else "[FAIL]"
            print(f"\n{status} Text: {text}")
            print(f"  Prediction: {result['label']} ({result['confidence']}% confidence)")
            print(f"  Probabilities: FAKE={result['probabilities'].get('FAKE', 0)}%, "
                  f"REAL={result['probabilities'].get('REAL', 0)}%")
        
        print("\n" + "-"*60)
        print("TESTING REAL NEWS SAMPLES:")
        print("-"*60)
        for text in real_samples:
            result = predictor.predict(text)
            status = "[PASS]" if result['label'] == 'REAL' else "[FAIL]"
            print(f"\n{status} Text: {text}")
            print(f"  Prediction: {result['label']} ({result['confidence']}% confidence)")
            print(f"  Probabilities: FAKE={result['probabilities'].get('FAKE', 0)}%, "
                  f"REAL={result['probabilities'].get('REAL', 0)}%")
        
        print("\n" + "="*60)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run 'python train.py' first to train the model.")


if __name__ == "__main__":
    main()
