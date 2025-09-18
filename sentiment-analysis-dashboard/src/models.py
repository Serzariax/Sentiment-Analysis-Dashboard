from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        self.model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

    def analyze(self, texts):
        results = self.model(texts)
        return results

    def get_confidence_scores(self, results):
        scores = []
        for result in results:
            scores.append(result['score'])
        return scores

    def classify_sentiment(self, text):
        result = self.model(text)[0]
        return result['label'], result['score']