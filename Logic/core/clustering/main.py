import time

import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import fasttext

from Logic.core.word_embedding.fasttext_data_loader import FastTextDataLoader
from Logic.core.word_embedding.fasttext_model import FastText
from Logic.core.clustering.dimension_reduction import DimensionReduction
from Logic.core.clustering.clustering_metrics import ClusteringMetrics
from Logic.core.clustering.clustering_utils import ClusteringUtils

# Main Function: Clustering Tasks
# 0. Embedding Extraction
# Using the previous preprocessor and fasttext model, collect all the embeddings of our data and store them.
loaded = np.load('../word_embedding/arrays.npz')
model = fasttext.load_model('../word_embedding/FastText_model.bin')
X = loaded['arr1']
y = loaded['arr2']
embeddings = []

for sentence in X:
    embedding = model.get_sentence_vector(sentence)
    embeddings.append(embedding)

embeddings = np.array(embeddings)

# 1. Dimension Reduction
# Perform Principal Component Analysis (PCA):
# Reduce the dimensionality of features using PCA. (you can use the reduced feature afterward or use to the whole embeddings)
# Find the Singular Values and use the explained_variance_ratio_ attribute to determine the percentage of variance explained by each principal component.
# Draw plots to visualize the results.

dimensionReduction = DimensionReduction()
dimensionReduction.wandb_plot_explained_variance_by_components(embeddings, 'clustering', 'explained_variance_by_components')

# Implement t-SNE (t-Distributed Stochastic Neighbor Embedding):
# Create the convert_to_2d_tsne function, which takes a list of embedding vectors as input and reduces the dimensionality to two dimensions using the t-SNE method.
# Use the output vectors from this step to draw the diagram.

dimensionReduction.wandb_plot_2d_tsne(embeddings, 'clustering', '2d_tsne')
reduced_embeddings = dimensionReduction.pca_reduce_dimension(embeddings, 2)

# 2. Clustering
## K-Means Clustering
# Implement the K-means clustering algorithm from scratch.
# Create document clusters using K-Means.
# Run the algorithm with several different values of k.
# For each run:
# Determine the genre of each cluster based on the number of documents in each cluster.
# Draw the resulting clustering using the two-dimensional vectors from the previous section.
# Check the implementation and efficiency of the algorithm in clustering similar documents.
# Draw the silhouette score graph for different values of k and perform silhouette analysis to choose the appropriate k.
# Plot the purity value for k using the labeled data and report the purity value for the final k. (Use the provided functions in utilities)

clusteringUtils = ClusteringUtils()
clusteringMetrics = ClusteringMetrics()

for k in range(2, 12):
    clusteringUtils.visualize_kmeans_clustering_wandb(reduced_embeddings, k, 'clustering', 'kmeans_clustering')

clusteringUtils.plot_kmeans_cluster_scores(reduced_embeddings, y, [k for k in range(2, 12)], 'clustering', 'kmeans_cluster_scores')
clusteringUtils.visualize_elbow_method_wcss(reduced_embeddings, [k for k in range(2, 12)], 'clustering', 'elbow_method_wcss')

## Hierarchical Clustering
# Perform hierarchical clustering with all different linkage methods.
# Visualize the results.

clusteringUtils.wandb_plot_hierarchical_clustering_dendrogram(reduced_embeddings, 'clustering', 'single', 'dendrogram-single')
clusteringUtils.wandb_plot_hierarchical_clustering_dendrogram(reduced_embeddings, 'clustering', 'complete', 'dendrogram-complete')
clusteringUtils.wandb_plot_hierarchical_clustering_dendrogram(reduced_embeddings, 'clustering', 'average', 'dendrogram-average')
clusteringUtils.wandb_plot_hierarchical_clustering_dendrogram(reduced_embeddings, 'clustering', 'ward', 'dendrogram-ward')


# 3. Evaluation
# Using clustering metrics, evaluate how well your clustering method is performing.

print('Kmeans')
_, labels = clusteringUtils.cluster_kmeans(reduced_embeddings, 4)
silhouette = clusteringMetrics.silhouette_score(reduced_embeddings, labels)
purity = clusteringMetrics.purity_score(y, labels)
ars = clusteringMetrics.adjusted_rand_score(y, labels)
print("Silhouette Score:", silhouette)
print("Purity Score:", purity)
print("Adjusted Rand Score:", ars)

print('Single')
labels = clusteringUtils.cluster_hierarchical_single(reduced_embeddings)
silhouette = clusteringMetrics.silhouette_score(reduced_embeddings, labels)
purity = clusteringMetrics.purity_score(y, labels)
ars = clusteringMetrics.adjusted_rand_score(y, labels)
print("Silhouette Score:", silhouette)
print("Purity Score:", purity)
print("Adjusted Rand Score:", ars)

print('Complete')
labels = clusteringUtils.cluster_hierarchical_complete(reduced_embeddings)
silhouette = clusteringMetrics.silhouette_score(reduced_embeddings, labels)
purity = clusteringMetrics.purity_score(y, labels)
ars = clusteringMetrics.adjusted_rand_score(y, labels)
print("Silhouette Score:", silhouette)
print("Purity Score:", purity)
print("Adjusted Rand Score:", ars)

print('Average')
labels = clusteringUtils.cluster_hierarchical_average(reduced_embeddings)
silhouette = clusteringMetrics.silhouette_score(reduced_embeddings, labels)
purity = clusteringMetrics.purity_score(y, labels)
ars = clusteringMetrics.adjusted_rand_score(y, labels)
print("Silhouette Score:", silhouette)
print("Purity Score:", purity)
print("Adjusted Rand Score:", ars)

print('Ward')
labels = clusteringUtils.cluster_hierarchical_ward(reduced_embeddings)
silhouette = clusteringMetrics.silhouette_score(reduced_embeddings, labels)
purity = clusteringMetrics.purity_score(y, labels)
ars = clusteringMetrics.adjusted_rand_score(y, labels)
print("Silhouette Score:", silhouette)
print("Purity Score:", purity)
print("Adjusted Rand Score:", ars)
