import numpy as np
import itertools
import random
import json


class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.num_hashes = num_hashes

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        shingles = set()
        tokens = document.split()
        for i in range(len(tokens) - 1):
            shingles.add(tokens[i] + " " + tokens[i + 1])
        return shingles

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        shingle_sets = [self.shingle_document(doc) for doc in self.documents]
        all_shingles = list(set().union(*shingle_sets))

        characteristic_matrix = np.zeros((len(shingle_sets), len(all_shingles)))

        for i in range(len(shingle_sets)):
            for j in range(len(all_shingles)):
                if all_shingles[j] in shingle_sets[i]:
                    characteristic_matrix[i, j] = 1

        return characteristic_matrix

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        characteristic_matrix = self.build_characteristic_matrix()
        signature = np.full((characteristic_matrix.shape[0], self.num_hashes), np.inf)
        hash_functions = []

        for _ in range(self.num_hashes):
            hash_functions.append(random.randint(0, len(self.documents) * 10))

        for i in range(len(self.documents)):
            shingles = self.shingle_document(self.documents[i])
            for j in range(len(hash_functions)):
                    for shingle in shingles:
                        shingle = " ".join(shingle)
                        hash = hash_functions[j]
                        for char in shingle:
                            hash = (ord(char) + hash) % (len(self.documents))

                        signature[i][j] = min(hash, signature[i][j])

        return signature

    def lsh_buckets(self, signature, bands=10, rows_per_band=10):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        buckets = {}

        for i, sig in enumerate(signature):
            for b in range(bands):
                h_band = hash(tuple(sig[b * rows_per_band:(b + 1) * rows_per_band]))
                buckets.setdefault(h_band, []).append(i)

        return buckets

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        signature = self.min_hash_signature()
        buckets = self.lsh_buckets(signature)
        return buckets

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        intersection = len(first_set.intersection(second_set))
        union = len(first_set.union(second_set))

        if union == 0:
            return 0.0

        return intersection / union

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print(correct_near_duplicates)
        print(all_near_duplicates)
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)


with open('D:/uni/term6/my/mir/project/mir_project/logic/core/LSHFakeData.json', 'r') as f:
    movies = json.load(f)

with open('D:/uni/term6/my/mir/project/mir_project/logic/IMDB_crawled(Without null).json', 'r') as f:
    all_movies = json.load(f)

docs = []

for movie in movies:
    if len(movie['summaries']) > 0:
        docs.append(' '.join(movie['summaries']))

for movie in all_movies:
    if len(movie['summaries']) > 0:
        docs.append(' '.join(movie['summaries']))

minHashLSH = MinHashLSH(docs, 100)
minHashLSH.jaccard_similarity_test(minHashLSH.perform_lsh(), docs)
