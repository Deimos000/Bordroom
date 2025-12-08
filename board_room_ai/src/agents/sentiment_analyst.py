from transformers import pipeline
import numpy as np

class SentimentAnalyst:
    def __init__(self, personality="neutral"):
        self.personality = personality # 'hype' or 'doom'
        # Using a smaller model or the one specified.
        # Note: In a real environment, this downloads the model.
        # I'll stick to the blueprint.
        try:
            self.nlp = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
        except Exception as e:
            print(f"Error loading FinBERT model: {e}")
            self.nlp = None

    def analyze_news(self, headlines, dossier):
        # 1. Filter headlines based on Dossier Keywords
        if self.nlp is None:
            return 0.0

        keywords_key = 'keywords_' + ('positive' if self.personality == 'hype' else 'negative')
        # The prompt used 'keywords_hype' / 'keywords_doom', but dossier has 'keywords_positive'/'negative'.
        # I will map hype->positive, doom->negative + macro_fear.

        keywords = []
        if self.personality == 'hype':
            keywords = dossier.get('keywords_positive', [])
        elif self.personality == 'doom':
            keywords = dossier.get('keywords_negative', []) + dossier.get('keywords_macro_fear', [])

        relevant_news = [h for h in headlines if any(k in h.lower() for k in keywords)]

        if not relevant_news:
            return 0.0

        # 2. Get Sentiment Score
        # Since pipeline might fail on large batch or weird input, we handle safely
        try:
            results = self.nlp(relevant_news)
        except Exception as e:
            print(f"Error in NLP processing: {e}")
            return 0.0

        # 3. Apply Personality Bias
        score = 0
        for res in results:
            if self.personality == "hype" and res['label'] == "Positive":
                score += res['score'] # Hype man amplifies good news
            elif self.personality == "doom" and res['label'] == "Negative":
                score -= res['score'] * 1.5 # Doomsayer amplifies bad news

        return score
