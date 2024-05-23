from .graph import LinkGraph
import networkx as nx
import json
import random


class LinkAnalyzer:
    def __init__(self, root_set):
        """
        Initialize the Link Analyzer attributes:

        Parameters
        ----------
        root_set: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "title": string of movie title
            "stars": A list of movie star names
        """
        self.root_set = root_set
        self.graph = LinkGraph()
        self.hubs = set()
        self.authorities = set()
        self.initiate_params()

    def initiate_params(self):
        """
        Initialize links graph, hubs list and authorities list based of root set

        Parameters
        ----------
        This function has no parameters. You can use self to get or change attributes
        """
        for movie in self.root_set:
            movie_id = movie["id"]
            self.graph.add_node(movie_id)
            self.hubs.add(movie_id)
            if movie['stars'] is not None:
                for star in movie["stars"]:
                    self.graph.add_edge(movie_id, star)
                    self.authorities.add(star)
    def expand_graph(self, corpus):
        """
        expand hubs, authorities and graph using given corpus

        Parameters
        ----------
        corpus: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "stars": A list of movie star names

        Note
        ---------
        To build the base set, we need to add the hubs and authorities that are inside the corpus
        and refer to the nodes in the root set to the graph and to the list of hubs and authorities.
        """
        for movie in corpus:
            movie_id = movie["id"]
            if movie["stars"] is not None:
                for star in movie["stars"]:
                    if any(star in root_movie["stars"] for root_movie in self.root_set):
                        self.graph.add_node(movie_id)
                        self.graph.add_edge(movie_id, star)
                        self.authorities.add(star)
                        self.hubs.add(movie_id)

    def hits(self, num_iteration=5, max_result=10):
        """
        Return the top movies and actors using the Hits algorithm

        Parameters
        ----------
        num_iteration: int
            Number of algorithm execution iterations
        max_result: int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            List of names of 10 actors with the most scores obtained by Hits algorithm in descending order
        list
            List of names of 10 movies with the most scores obtained by Hits algorithm in descending order
        """
        a_s = []
        h_s = []
        h, a = nx.hits(self.graph.graph, max_iter=num_iteration)
        sorted_hubs = sorted(h, key=h.get, reverse=True)[:max_result]
        sorted_authorities = sorted(a, key=a.get, reverse=True)

        for sorted_authority in sorted_authorities:
            if len(a_s) == max_result:
                break
            if not sorted_authority.startswith('tt'):
                a_s.append(sorted_authority)

        for sorted_hub in sorted_hubs:
            if len(h_s) == max_result:
                break
            if sorted_hub.startswith('tt'):
                h_s.append(sorted_hub)
        return a_s, h_s


if __name__ == "__main__":
    # You can use this section to run and test the results of your link analyzer
    with open('../../IMDB_crawled.json', 'r') as f:
        corpus = json.load(f)

    root_set = [corpus[0], corpus[50], corpus[200], corpus[100], corpus[1000], corpus[700], corpus[300]]

    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=5)
    print("Top Actors:")
    print(*actors, sep=' - ')
    print("Top Movies:")
    print(*movies, sep=' - ')
