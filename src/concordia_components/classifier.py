
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class ActionClassifier:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model_path = model_path
        # Define the labels explicitly as per standard social media actions
        self.labels = ["Retweet", "Reply", "Like", "Quote", "Ignore"]
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Warning: Could not load BERT model from {model_path}. Using dummy classifier. Error: {e}")
            self.model = None

    def predict_batch(self, tweets: list[str]) -> list[str]:
        """
        Predicts the action to take for a batch of tweets.
        Returns a list of actions (str).
        """
        if self.model is None:
            # Dummy fallback if model fails to load (for testing/safety)
            return np.random.choice(self.labels, size=len(tweets))

        inputs = self.tokenizer(tweets, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        return [self.labels[p.item()] for p in predictions]
