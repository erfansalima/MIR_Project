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

        doc_words = doc.split()
        query_words = self.remove_stop_words_from_query(query).split()

        for query_word in query_words:

            if query_word not in doc_words:
                not_exist_words.append(query_word)
            else:
                for i in range(len(doc_words)):
                    if doc_words[i] == query_word:
                        occurrence = i
                        break

                start = max(0, occurrence - self.number_of_words_on_each_side)
                end = min(len(doc_words), occurrence + self.number_of_words_on_each_side + 1)
                snippet_words = doc_words[start:end]

                snippet_words[snippet_words.index(query_word)] = f"***{query_word}***"
                final_snippet += " ".join(snippet_words) + " ... "

        return final_snippet[:len(final_snippet)-5], not_exist_words

