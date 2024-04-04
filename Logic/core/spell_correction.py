import nltk

class SpellCorrection:
    def __init__(self, all_documents):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str
            The input documents.
        """
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(all_documents)

    def shingle_word(self, word, k=2):
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """
        shingles = set()

        if len(word) < k:
            shingles.add(word)
        else:
            for i in range(len(word) - k + 1):
                shingles.add(word[i:i + k])
        return shingles
    
    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """
        intersection = len(first_set.intersection(second_set))
        union = len(first_set.union(second_set))
        if union == 0:
            return 0.0
        return intersection / union

    def shingling_and_counting(self, all_documents):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_shingled_words = dict()
        word_counter = dict()

        for doc in all_documents:
            for word in doc.split():
                if word not in all_shingled_words:
                    word_counter[word] = 0
                    all_shingled_words[word] = self.shingle_word(word)
                word_counter[word] += 1
                
        return all_shingled_words, word_counter
    
    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : stf
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        shingled_word = self.shingle_word(word)
        similarity_scores = {}

        for candidate_word, candidate_shingles in self.all_shingled_words.items():
            score = self.jaccard_score(shingled_word, candidate_shingles)
            similarity_scores[candidate_word] = score

        sorted_candidates = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
        return [candidate[0] for candidate in sorted_candidates[:5]]

    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : stf
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        corrected_query = []

        for word in nltk.word_tokenize(query.lower()):
            if word in self.word_counter:
                corrected_query.append(word)
            else:
                nearest_words = self.find_nearest_words(word)
                if nearest_words:
                    word_shingles = self.shingle_word(word)
                    corrected_query.append(max(nearest_words, key=lambda word: self.word_counter[word] * self.jaccard_score(word_shingles, self.shingle_word(word))))
                else:
                    corrected_query.append(word)

        return ' '.join(corrected_query)
