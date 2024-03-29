import nltk
from nltk.corpus import stopwords


class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

    def remove_stop_words_from_query(self, query):
        """
        Remove stop words from the input string.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        str
            The query without stop words.
        """

        words = query.split()
        query_without_stopwords = ' '.join([word for word in words if word.lower() not in self.stop_words])
        return query_without_stopwords

    def find_snippet(self, doc, query):
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Sahwshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """
        final_snippet = ""
        not_exist_words = []

        doc_words = set(doc.lower().split())
        query_words = set(self.remove_stop_words_from_query(query).lower().split())
        exist_words = doc_words.intersection(query_words)
        not_exist_words = list(query_words - exist_words)

        final_snippet = doc
        for word in exist_words:
            word_index = doc.lower().index(word)
            start_index = max(0, word_index - self.number_of_words_on_each_side * 2)
            end_index = min(len(doc), word_index + self.number_of_words_on_each_side * 2)
            snippet_part = doc[start_index:end_index]
            final_snippet = final_snippet.replace(snippet_part, f'***{snippet_part}***')

        return final_snippet, not_exist_words

        return final_snippet, not_exist_words
