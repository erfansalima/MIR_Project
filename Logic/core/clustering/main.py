import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from ..word_embedding.fasttext_data_loader import FastTextDataLoader
from ..word_embedding.fasttext_model import FastText
from .dimension_reduction import DimensionReduction
from .clustering_metrics import ClusteringMetrics
from .clustering_utils import ClusteringUtils

# Main Function: Clustering Tasks

# 0. Embedding Extraction
# TODO: Using the previous preprocessor and fasttext model, collect all the embeddings of our data and store them.
embeddings = FastText.get_query_embedding()
# 1. Dimension Reduction
# TODO: Perform Principal Component Analysis (PCA):
#     - Reduce the dimensionality of features using PCA. (you can use the reduced feature afterward or use to the whole embeddings)
#     - Find the Singular Values and use the explained_variance_ratio_ attribute to determine the percentage of variance explained by each principal component.
#     - Draw plots to visualize the results.
pca_result = DimensionReduction.perform_pca(embeddings)
DimensionReduction.visualize_pca_results(pca_result)

# TODO: Implement t-SNE (t-Distributed Stochastic Neighbor Embedding):
#     - Create the convert_to_2d_tsne function, which takes a list of embedding vectors as input and reduces the dimensionality to two dimensions using the t-SNE method.
#     - Use the output vectors from this step to draw the diagram.
tsne_result = DimensionReduction.convert_to_2d_tsne(embeddings)
DimensionReduction.visualize_tsne_results(tsne_result)

# 2. Clustering
## K-Means Clustering
# TODO: Implement the K-means clustering algorithm from scratch.
# TODO: Create document clusters using K-Means.
# TODO: Run the algorithm with several different values of k.
# TODO: For each run:
#     - Determine the genre of each cluster based on the number of documents in each cluster.
#     - Draw the resulting clustering using the two-dimensional vectors from the previous section.
#     - Check the implementation and efficiency of the algorithm in clustering similar documents.
# TODO: Draw the silhouette score graph for different values of k and perform silhouette analysis to choose the appropriate k.
# TODO: Plot the purity value for k using the labeled data and report the purity value for the final k. (Use the provided functions in utilities)
k_values = [2, 3, 4, 5]
kmeans_results = {}
for k in k_values:
    kmeans_clusters = ClusteringUtils.kmeans_clustering(embeddings, k)
    kmeans_results[k] = kmeans_clusters
    ClusteringUtils.visualize_clusters(kmeans_clusters)

silhouette_scores = ClusteringMetrics.calculate_silhouette_scores(embeddings, kmeans_results)
ClusteringMetrics.plot_silhouette_scores(silhouette_scores)
## Hierarchical Clustering
# TODO: Perform hierarchical clustering with all different linkage methods.
# TODO: Visualize the results.
linkage_methods = ['single', 'complete', 'average', 'ward']
hierarchical_results = {}

for method in linkage_methods:
    linkage_matrix = ClusteringUtils.wandb_plot_hierarchical_clustering_dendrogram(embeddings, linkage_method=method)
    hierarchical_results[method] = linkage_matrix

# 3. Evaluation
# TODO: Using clustering metrics, evaluate how well your clustering method is performing.
silhouette_scores = ClusteringMetrics.silhouette_score(embeddings, hierarchical_results)
purity_scores = ClusteringMetrics.purity_score(hierarchical_results, hierarchical_results)

print("Hierarchical Clustering Silhouette Scores:", silhouette_scores)
print("Hierarchical Clustering Purity Scores:", purity_scores)
