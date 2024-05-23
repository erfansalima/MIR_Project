import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

from Logic.core.indexer.index_reader import Index_reader
from Logic.core.indexer.indexes_enum import Indexes


class FastTextDataLoader:
    """
    This class is designed to load and pre-process data for training a FastText model.

    It takes the file path to a data source containing movie information (synopses, summaries, reviews, titles, genres) as input.
    The class provides methods to read the data into a pandas DataFrame, pre-process the text data, and create training data (features and labels)
    """
    def __init__(self, file_path):
        """
        Initializes the FastTextDataLoader class with the file path to the data source.

        Parameters
        ----------
        file_path: str
            The path to the file containing movie information.
        """
        self.file_path = file_path

    def read_data_to_df(self):
        """
        Reads data from the specified file path and creates a pandas DataFrame containing movie information.

        You can use an IndexReader class to access the data based on document IDs.
        It extracts synopses, summaries, reviews, titles, and genres for each movie.
        The extracted data is then stored in a pandas DataFrame with appropriate column names.

        Returns
        ----------
            pd.DataFrame: A pandas DataFrame containing movie information (synopses, summaries, reviews, titles, genres).
        """
        data = Index_reader(self.file_path, index_name=Indexes.DOCUMENTS).index

        # Extract required information (synopses, summaries, reviews, titles, genres)
        synopses = []
        summaries = []
        reviews = []
        titles = []
        genres = []

        for movie_id, movie_data in tqdm(data.items()):
            synopses.append(movie_data.get('synopsis', ''))
            summaries.append(movie_data.get('summaries', ''))
            reviews.append(movie_data.get('reviews', ''))
            titles.append(movie_data.get('title', ''))
            genres.append(movie_data.get('genres', ''))

        # Create DataFrame
        df = pd.DataFrame({
            'synopsis': synopses,
            'summary': summaries,
            'reviews': reviews,
            'title': titles,
            'genre': genres
        })
        return df

    def preprocess_text(self, text):
        """
        Preprocesses the text data by lowercasing, removing punctuation, stopwords, and tokenizing.

        Parameters
        ----------
        text: str
            The text to preprocess.

        Returns
        -------
        str
            The preprocessed text.
        """
        x = ''
        if text is not None:
            for text in text:
                text = ' '.join(text)
                text = text.lower()
                text = text.translate(str.maketrans('', '', string.punctuation))
                tokens = word_tokenize(text)
                stop_words = set(stopwords.words('english'))
                filtered_tokens = [token for token in tokens if token not in stop_words]
                preprocessed_text = ' '.join(filtered_tokens)
                x += preprocessed_text

        return x

    def create_train_data(self):
        """
        Reads data using the read_data_to_df function, pre-processes the text data, and creates training data (features and labels).

        Returns:
            tuple: A tuple containing two NumPy arrays: X (preprocessed text data) and y (encoded genre labels).
        """
        df = self.read_data_to_df()

        for col in ['reviews']:
            df[f'preprocessed_{col}'] = df[col].apply(self.preprocess_text)

        label_encoder = LabelEncoder()
        first_genre = []

        # Iterate over each element in the 'genre' column and extract the first character
        for genre in df['genre']:
            if len(genre) > 0:
                first_genre.append(genre[0])
            else:
                first_genre.append('nothing')

        df['encode_genre'] = first_genre
        df['encoded_genre'] = label_encoder.fit_transform(df['encode_genre'].astype(str))
        X = df['preprocessed_reviews'].values.astype(str)
        y = df['encoded_genre'].values
        X = X[:1000]
        y = y[:1000]
        print(y)

        return X, y
