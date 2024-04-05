import time
import os
import json
import copy
from indexes_enum import Indexes
from tiered_index import Tiered_index


class Index:
    def __init__(self, preprocessed_documents: list):
        """
        Create a class for indexing.
        """

        self.preprocessed_documents = preprocessed_documents
        self.terms = []
        self.index = {
            Indexes.DOCUMENTS.value: self.index_documents(),
            Indexes.STARS.value: self.index_stars(),
            Indexes.GENRES.value: self.index_genres(),
            Indexes.SUMMARIES.value: self.index_summaries(),
        }

    def index_documents(self):
        """
        Index the documents based on the document ID. In other words, create a dictionary
        where the key is the document ID and the value is the document.

        Returns
        ----------
        dict
            The index of the documents based on the document ID.
        """

        current_index = {}

        for document in self.preprocessed_documents:
            document_id = document['id']
            current_index[document_id] = document

        return current_index

    def index_stars(self):
        """
        Index the documents based on the stars.

        Returns
        ----------
        dict
            The index of the documents based on the stars. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """
        star_index = {}

        for document in self.preprocessed_documents:
            document_id = document['id']
            stars = document['stars']
            names = [word.lower() for name in stars for word in name.split()]
            for star in stars:
                star = star.lower()
                parts = star.split()
                for part in parts:
                    self.terms.append(part)
                    if part not in star_index:
                        star_index[part] = {}
                    star_index[part][document_id] = names.count(part)

        return star_index

    def index_genres(self):
        """
        Index the documents based on the genres.

        Returns
        ----------
        dict
            The index of the documents based on the genres. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """
        genre_index = {}

        for document in self.preprocessed_documents:
            document_id = document['id']
            genres = document['genres']
            for genre in genres:
                if genre not in genre_index:
                    self.terms.append(genre.lower())
                    genre_index[genre.lower()] = {}
                genre_index[genre.lower()][document_id] = genres.count(genre)

        return genre_index

    def index_summaries(self):
        """
        Index the documents based on the summaries (not first_page_summary).

        Returns
        ----------
        dict
            The index of the documents based on the summaries. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """
        index = {}

        for doc in self.preprocessed_documents:
            doc_id = doc['id']
            summaries = doc.get('summaries', [])
            for summary in summaries:
                terms = summary.split()
                term_freq = {}
                for term in terms:
                    term_freq[term] = term_freq.get(term, 0) + 1
                for term, tf in term_freq.items():
                    if term not in index:
                        self.terms.append(term)
                        index[term] = {}
                    index[term][doc_id] = tf
        return index

    def get_posting_list(self, word: str, index_type: str):
        """
        get posting_list of a word

        Parameters
        ----------
        word: str
            word we want to check
        index_type: str
            type of index we want to check (documents, stars, genres, summaries)

        Return
        ----------
        list
            posting list of the word (you should return the list of document IDs that contain the word and ignore the tf)
        """
        try:
            posting_list = list(self.index[index_type].get(word, {}).keys())
            return posting_list
        except Exception as e:
            print(f"Error occurred: {e}")
            return []

    def add_document_to_index(self, document: dict):
        """
        Add a document to all the indexes

        Parameters
        ----------
        document : dict
            Document to add to all the indexes
        """
        document_id = document['id']

        self.index[Indexes.DOCUMENTS.value][document_id] = document

        stars = document.get('stars', [])
        for star in stars:
            if star not in self.index[Indexes.STARS.value]:
                self.index[Indexes.STARS.value][star] = {}
            self.index[Indexes.STARS.value][star][document_id] = stars.count(star)

        genres = document.get('genres', [])
        for genre in genres:
            if genre not in self.index[Indexes.GENRES.value]:
                self.index[Indexes.GENRES.value][genre] = {}
            self.index[Indexes.GENRES.value][genre][document_id] = genres.count(genre)

        summaries = document.get('summaries', [])
        for summary in summaries:
            terms = summary.split()
            for term in terms:
                if term not in self.index[Indexes.SUMMARIES.value]:
                    self.index[Indexes.SUMMARIES.value][term] = {}
                if document_id not in self.index[Indexes.SUMMARIES.value][term]:
                    self.index[Indexes.SUMMARIES.value][term][document_id] = terms.count(term)

    def remove_document_from_index(self, document_id: str):
        """
        Remove a document from all the indexes

        Parameters
        ----------
        document_id : str
            ID of the document to remove from all the indexes
        """

        if document_id in self.index['documents']:
            del self.index['documents'][document_id]

        keys_to_delete = []
        for x in self.index['stars']:
            for movie in self.index['stars'][x]:
                if document_id == movie:
                    keys_to_delete.append(x)
                    break

        for key in keys_to_delete:
            del self.index['stars'][key][document_id]

        keys_to_delete = []
        for x in self.index['genres']:
            for movie in self.index['genres'][x]:
                if document_id == movie:
                    keys_to_delete.append(x)
                    break

        for key in keys_to_delete:
            del self.index['genres'][key][document_id]

        keys_to_delete = []
        for x in self.index['summaries']:
            for movie in self.index['summaries'][x]:
                if document_id == movie:
                    keys_to_delete.append(x)
                    break

        for key in keys_to_delete:
            del self.index['summaries'][key][document_id]

    def check_add_remove_is_correct(self):
        """
        Check if the add and remove is correct
        """

        dummy_document = {
            'id': '100',
            'stars': ['tim', 'henry'],
            'genres': ['drama', 'crime'],
            'summaries': ['good']
        }

        index_before_add = copy.deepcopy(self.index)
        self.add_document_to_index(dummy_document)
        index_after_add = copy.deepcopy(self.index)

        if index_after_add[Indexes.DOCUMENTS.value]['100'] != dummy_document:
            print('Add is incorrect, document')
            return

        if (set(index_after_add[Indexes.STARS.value]['tim']).difference(
                set(index_before_add[Indexes.STARS.value]['tim']))
                != {dummy_document['id']}):
            print('Add is incorrect, tim')
            return

        if (set(index_after_add[Indexes.STARS.value]['henry']).difference(
                set(index_before_add[Indexes.STARS.value]['henry']))
                != {dummy_document['id']}):
            print('Add is incorrect, henry')
            return
        if (set(index_after_add[Indexes.GENRES.value]['drama']).difference(
                set(index_before_add[Indexes.GENRES.value]['drama']))
                != {dummy_document['id']}):
            print('Add is incorrect, drama')
            return

        if (set(index_after_add[Indexes.GENRES.value]['crime']).difference(
                set(index_before_add[Indexes.GENRES.value]['crime']))
                != {dummy_document['id']}):
            print('Add is incorrect, crime')
            return

        if (set(index_after_add[Indexes.SUMMARIES.value]['good']).difference(
                set(index_before_add[Indexes.SUMMARIES.value]['good']))
                != {dummy_document['id']}):
            print('Add is incorrect, good')
            return

        print('Add is correct')

        self.remove_document_from_index('100')
        index_after_remove = copy.deepcopy(self.index)

        if index_after_remove == index_before_add:
            print('Remove is correct')
        else:
            print('Remove is incorrect')

    def store_index(self, path: str, index_type: str = None):
        """
        Stores the index in a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to store the file
        index_type: str or None
            type of index we want to store (documents, stars, genres, summaries)
            if None store tiered index
        """

        if not os.path.exists(path):
            os.makedirs(path)

        if index_type is None:
            tiered = Tiered_index(path="index/")

        elif index_type not in self.index:
            raise ValueError('Invalid index type')

        else:
            index_to_store = self.index[index_type]

            with open(os.path.join(path, index_type + '_' + 'index.json'), 'w') as f:
                json.dump(index_to_store, f)

    def load_index(self, path: str):
        """
        Loads the index from a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to load the file
        """
        with open(os.path.join(path, 'index.json'), 'r') as f:
            loaded_index = json.load(f)

        self.index.update(loaded_index)

    def check_if_index_loaded_correctly(self, index_type: str, loaded_index: dict):
        """
        Check if the index is loaded correctly

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        loaded_index : dict
            The loaded index

        Returns
        ----------
        bool
            True if index is loaded correctly, False otherwise
        """

        return self.index[index_type] == loaded_index

    def check_if_indexing_is_good(self, index_type: str, check_word: str = 'good'):
        """
        Checks if the indexing is good. Do not change this function. You can use this
        function to check if your indexing is correct.

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        check_word : str
            The word to check in the index

        Returns
        ----------
        bool
            True if indexing is good, False otherwise
        """

        # brute force to check check_word in the summaries
        start = time.time()
        docs = []
        for document in self.preprocessed_documents:
            if index_type not in document or document[index_type] is None:
                continue

            for field in document[index_type]:
                if check_word in field:
                    docs.append(document['id'])
                    break

            # if we have found 3 documents with the word, we can break
            if len(docs) == 3:
                break

        end = time.time()
        brute_force_time = end - start

        # check by getting the posting list of the word
        start = time.time()
        # TODO: based on your implementation, you may need to change the following line
        posting_list = self.get_posting_list(check_word, index_type)

        end = time.time()
        implemented_time = end - start

        print('Brute force time: ', brute_force_time)
        print('Implemented time: ', implemented_time)

        if set(docs).issubset(set(posting_list)):
            print('Indexing is correct')

            if implemented_time < brute_force_time:
                print('Indexing is good')
                return True
            else:
                print('Indexing is bad')
                return False
        else:
            print('Indexing is wrong')
            return False


# TODO: Run the class with needed parameters, then run check methods and finally report the results of check methods
with open('D:\\uni\\term6\\MY\\MIR\\Project\\MIR_Project\\Logic\\IMDB_crawled(Without null).json', 'r') as f:
    movies_dataset = json.load(f)

index = Index(movies_dataset)
index.store_index('D:\\uni\\term6\\MY\\MIR\\Project\\MIR_Project\\Logic\\core\\indexer\\index')
# index.store_index('D:\\uni\\term6\\MY\\MIR\\Project\\MIR_Project\\Logic\\core\\indexer\\index', index_type=Indexes.DOCUMENTS.value)
# index.store_index('D:\\uni\\term6\\MY\\MIR\\Project\\MIR_Project\\Logic\\core\\indexer\\index', index_type=Indexes.STARS.value)
# index.store_index('D:\\uni\\term6\\MY\\MIR\\Project\\MIR_Project\\Logic\\core\\indexer\\index', index_type=Indexes.GENRES.value)
# index.store_index('D:\\uni\\term6\\MY\\MIR\\Project\\MIR_Project\\Logic\\core\\indexer\\index', index_type=Indexes.SUMMARIES.value)
# index.check_add_remove_is_correct()

with open('../terms.json', 'w') as f:
    json.dump(index.terms, f)
