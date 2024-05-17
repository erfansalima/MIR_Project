import numpy as np
from tqdm import tqdm

# from word_embedding.fasttext_model import FastText


class BasicClassifier:
    def __init__(self):
        self.x = None

    def fit(self, x, y):
        self.x = None

    def predict(self, x):
        self.x = None

    def prediction_report(self, x, y):
        self.x = None

    def get_percent_of_positive_reviews(self, sentences):
        """
        Get the percentage of positive reviews in the given sentences
        Parameters
        ----------
        sentences: list
            The list of sentences to get the percentage of positive reviews
        Returns
        -------
        float
            The percentage of positive reviews
        """
        positive_count = 0
        for sentence in sentences:
            prediction = self.predict([sentence])[0]
            if prediction == 'positive':
                positive_count += 1
        return positive_count / len(sentences)

