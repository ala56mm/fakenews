"""
Text Preprocessing Module for Fake News Detection System
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


class TextPreprocessor:
    """Handles text preprocessing operations"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """
        Apply all preprocessing steps to text
        
        Steps:
        1. Convert to lowercase
        2. Remove punctuation
        3. Remove numbers
        4. Remove stopwords
        5. Lemmatize words
        6. Remove extra whitespace
        """
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


if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    
    samples = [
        "BREAKING: Scientists DISCOVERED a MIRACLE cure!!! Don't miss this...",
        "Government announces new policy changes for 2024",
        "Amazing weight loss secret revealed - lose 30 lbs in one week!"
    ]
    
    print("TEXT PREPROCESSING TEST")
    print("="*60)
    for text in samples:
        print(f"\nOriginal: {text}")
        print(f"Processed: {preprocessor.clean_text(text)}")
