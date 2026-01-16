
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from concordia.language_model.language_model import LanguageModel

class ActionClassifier:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model_path = model_path
        # The full list of labels the BERT classifier was trained on
        self.raw_labels = [
            "like", "unlike", "repost", "unrepost", "follow", "unfollow",
            "block", "unblock", "post_update", "post_delete", "quote", "post", "reply"
        ]
        
        # Mapping to the actions supported by our simulation engine
        self.action_mapping = {
            "like": "Like",
            "repost": "Repost",
            "quote": "Quote",
            "reply": "Reply"
        }
        
        # try:
        #     self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        #     self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        #     self.model.to(self.device)
        #     self.model.eval()
        # except Exception as e:
        e = Exception("Model loading skipped for this example.")
        print(f"Warning: Could not load BERT model from {model_path}. Using dummy classifier. Error: {e}")
        self.model = None

    def predict_batch(self, tweets: list[str]) -> list[str]:
        """
        Predicts the action to take for a batch of tweets.
        Returns a list of actions (str).
        """
        if self.model is None:
            # Dummy fallback if model fails to load (for testing/safety)
            return [np.random.choice(["Like", "Repost", "Reply", "Quote", "Ignore"]) for _ in range(len(tweets))]

        inputs = self.tokenizer(tweets, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Map raw BERT labels to engine actions, default to "Ignore"
        results = []
        for p in predictions:
            raw_label = self.raw_labels[p.item()]
            mapped_action = self.action_mapping.get(raw_label, "Ignore")
            results.append(mapped_action)
            
        return results

class MinitaurClassifier:
    def __init__(self, model: LanguageModel):
        self.model = model
        self.action_keys = {
            "A": "Like",
            "B": "Repost",
            "C": "Quote",
            "D": "Reply",
            "E": "Ignore"
        }

    def predict_batch(self, tweets: list[str]) -> list[str]:
        results = []
        choices = ["A", "B", "C", "D", "E"]
        for tweet in tweets:
            prompt = (
                "You will be presented with a tweet and 5 possible actions.\n"
                "Please indicate which action you would like to take specific to the tweet by pressing the corresponding key.\n\n"
                f"Tweet: {tweet}\n"
                "A: Like, B: Repost, C: Quote, D: Reply, E: Ignore. You press <<"
            )
            
            # Use sample_choice to pick the most likely completion
            idx, response, _ = self.model.sample_choice(prompt=prompt, responses=choices)
            
            # The response is e.g. "A". The key is response[0]
            key = response[0]
            results.append(self.action_keys.get(key, "Ignore"))
            
        return results
