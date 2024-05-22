import json
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

class Preprocessor:

    def __init__(self, documents: list):
        """
        Initialize the class.

        Parameters
        ----------
        documents : list
            The list of documents to be preprocessed, path to stop words, or other parameters.
        """
        self.documents = documents
        self.stopwords = set(stopwords.words('english'))

    def preprocess(self):
        """
        Preprocess the text using the methods in the class.

        Returns
        ----------
        List[str]
            The preprocessed documents.
        """
    #    preprocessed_documents = []
    #    for doc in self.documents:
    #        preprocessed_doc = {}
    #        for key, value in doc.items():
    #            if key in ['first_page_summary', 'summaries', 'synposis', 'reviews', 'writers', 'stars', 'directors']:
    #                preprocessed_doc[key] = self.preprocess_attribute(value)
    #            else:
    #                preprocessed_doc[key] = doc[key]
    #            preprocessed_documents.append(preprocessed_doc)
    #    return preprocessed_documents

        preprocessed_documents = []
        for document in self.documents:
            preprocessed_value = self.normalize(document)
            preprocessed_value = self.remove_links(preprocessed_value)
            preprocessed_value = self.remove_punctuations(preprocessed_value)
            preprocessed_value = self.tokenize(preprocessed_value)
            preprocessed_value = self.remove_stopwords(preprocessed_value)
            preprocessed_documents.append(preprocessed_value)
        return preprocessed_documents

    def preprocess_attribute(self, attribute):
        """
        Preprocess a single attribute.

        Parameters
        ----------
        attribute : str or List[str] or None
            The attribute value to be preprocessed.

        Returns
        ----------
        str or List[str] or None
            The preprocessed attribute value.
        """
        if isinstance(attribute, str):
            attribute = self.normalize(attribute)
            attribute = self.remove_links(attribute)
            attribute = self.remove_punctuations(attribute)
            attribute = self.tokenize(attribute)
            attribute = self.remove_stopwords(attribute)
        elif isinstance(attribute, list):
            attribute = [self.preprocess_attribute(item) for item in attribute]
        return attribute

    def normalize(self, text: str):
        """
        Normalize the text by converting it to a lower case, stemming, lemmatization, etc.

        Parameters
        ----------
        text : str
            The text to be normalized.

        Returns
        ----------
        str
            The normalized text.
        """
        lemmatizer = WordNetLemmatizer()
        normalized_word = lemmatizer.lemmatize(text)
        normalized_word = re.sub(r'\b\w*\d\w*\b', '', normalized_word)
        return normalized_word.lower()

    def remove_links(self, text: str):
        """
        Remove links from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with links removed.
        """
        patterns = [r'\S*http\S*', r'\S*www\S*', r'\S+\.ir\S*', r'\S+\.com\S*', r'\S+\.org\S*', r'\S*@\S*']

        for pattern in patterns:
            text = re.sub(pattern, '', text)
        return text

    def remove_punctuations(self, text: str):
        """
        Remove punctuations from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with punctuations removed.
        """
        return re.sub(r'[^\w\s]', '', text)

    def tokenize(self, text: str):
        """
        Tokenize the words in the text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        ----------
        list
            The list of words.
        """
        return word_tokenize(text)

    def remove_stopwords(self, tokens: str):
        """
        Remove stopwords from the text.

        Parameters
        ----------
        text : str
            The text to remove stopwords from.

        Returns
        ----------
        list
            The list of words with stopwords removed.
        """
        filtered_tokens = [token for token in tokens if token.lower() not in self.stopwords and not any(char.isdigit() for char in token)]
        filtered_tokens = ' '.join(filtered_tokens)
        return filtered_tokens


#with open('../IMDB_crawled(preprocessed).json', 'r') as f:
#    x = json.load(f)

#preprocessor = Preprocessor(x)
#prep = preprocessor.preprocess()

#with open('../IMDB_crawled(preprocessed).json', 'w') as f:
#    json.dump(prep, f, indent=4)


