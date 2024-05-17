import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm

from basic_classifier import BasicClassifier
from data_loader import ReviewLoader


class KnnClassifier(BasicClassifier):
    def __init__(self, n_neighbors):
        super().__init__()
        self.k = n_neighbors
        self.x_train = None
        self.y_train = None

    def fit(self, x, y):
        """
        Fit the model using X as training data and y as target values.

        Parameters:
        ----------
        x: np.ndarray
            An m * n matrix - m is the count of documents and n is the embedding size.
        y: np.ndarray
            The real class label for each document.

        Returns:
        -------
        self
            Returns self as a fitted classifier.
        """
        self.x_train = x
        self.y_train = y
        return self

    def predict(self, x):
        """
        Predict the class label for each document in x using the fitted KNN model.

        Parameters:
        ----------
        x: np.ndarray
            An k * n matrix - k is the count of documents and n is the embedding size.

        Returns:
        -------
        np.ndarray
            An array containing the predicted class labels for each document in x.
        """
        predictions = []
        for sample in x:
            distances = np.linalg.norm(self.x_train - sample, axis=1)
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_indices]
            unique, counts = np.unique(nearest_labels, return_counts=True)
            predicted_label = unique[np.argmax(counts)]
            predictions.append(predicted_label)
        return np.array(predictions)

    def prediction_report(self, x, y):
        """
        Generate a classification report based on the predictions for x and the true labels y.

        Parameters:
        ----------
        x: np.ndarray
            An k * n matrix - k is the count of documents and n is the embedding size.
        y: np.ndarray
            The real class label for each document.

        Returns:
        -------
        str
            The classification report as a string.
        """
        return classification_report(x, y)


# F1 Accuracy : 70%
if __name__ == '__main__':
    data_loader = ReviewLoader('IMDBDataset.csv')
    data_loader.load_data()
    X_train, X_test, y_train, y_test = data_loader.split_data()
    knn_classifier = KnnClassifier(n_neighbors=5)
    knn_classifier.fit(X_train, y_train)
    report = knn_classifier.prediction_report(knn_classifier.predict(X_test), y_test)
    print("Classification Report:")
    print(report)
