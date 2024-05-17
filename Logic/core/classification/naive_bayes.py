import numpy as np
import pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from basic_classifier import BasicClassifier
from data_loader import ReviewLoader
from Logic.core.word_embedding.fasttext_model import preprocess_text


class NaiveBayes(BasicClassifier):
    def __init__(self, count_vectorizer, alpha=1):
        super().__init__()
        self.cv = count_vectorizer
        self.alpha = alpha
        self.num_classes = None
        self.classes = None
        self.number_of_features = None
        self.number_of_samples = None
        self.prior = None
        self.feature_probabilities = None
        self.log_probs = None

    def fit(self, x, y):
        """
        Fit the features and the labels
        Calculate prior and feature probabilities

        Parameters:
        ----------
        x: np.ndarray
            An m * n matrix - m is the number of documents and n is the vocabulary size (number of unique words)

        y: np.ndarray
            The real class label for each document

        Returns:
        -------
        self
            Returns self as a classifier after fitting
        """
        self.number_of_samples, self.number_of_features = x.shape
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        self.prior = np.zeros(self.num_classes)

        for idx, label in enumerate(self.classes):
            self.prior[idx] = np.sum(y == label) / self.number_of_samples

        self.feature_probabilities = np.zeros((self.num_classes, self.number_of_features))
        for idx, label in enumerate(self.classes):
            x_label = x[y == label]
            self.feature_probabilities[idx, :] = (x_label.sum(axis=0) + self.alpha) / (
                        x_label.sum() + self.alpha * self.number_of_features)

        self.log_probs = np.log(self.feature_probabilities)

        return self

    def predict(self, x):
        """
        Parameters:
        ----------
        x: np.ndarray
            An k * n matrix - k is the number of documents to predict and n is the vocabulary size

        Returns:
        -------
        np.ndarray
            Return the predicted class for each document
            with the highest probability (argmax)
        """
        log_prior = np.log(self.prior)
        log_likelihood = x @ self.log_probs.T
        log_posterior = log_prior + log_likelihood
        return self.classes[np.argmax(log_posterior, axis=1)]

    def prediction_report(self, x, y):
        """
        Parameters:
        ----------
        x: np.ndarray
            An k * n matrix - k is the number of documents and n is the vocabulary size

        y: np.ndarray
            The real class label for each document

        Returns:
        -------
        str
            Return the classification report
        """
        y_pred = self.predict(x)
        return classification_report(y, y_pred)

    def get_percent_of_positive_reviews(self, sentences):
        """
        Override this method because we are using a different embedding method in this class.
        """
        x = self.cv.transform(sentences)
        y_pred = self.predict(x.toarray())
        positive_reviews = np.sum(y_pred == 1)
        return positive_reviews / len(sentences)


if __name__ == '__main__':
    """
    First, find the embeddings of the reviews using the CountVectorizer, then fit the model with the training data.
    Finally, predict the test data and print the classification report
    You can use scikit-learn's CountVectorizer to find the embeddings.
    """
    df = pandas.read_csv('IMDBDataset.csv')
    reviews = [preprocess_text(text.split()) for text in df['review']]

    sentiments = df['sentiment'].values
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(sentiments)

    count_vectorizer = CountVectorizer(max_features=5000)
    X = count_vectorizer.fit_transform(reviews).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, sentiments, test_size=0.2, random_state=42)

    nb_classifier = NaiveBayes(count_vectorizer)
    nb_classifier.fit(X_train, y_train)

    print(nb_classifier.prediction_report(X_test, y_test))
