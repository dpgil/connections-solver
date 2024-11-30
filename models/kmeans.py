from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from typing import List

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np


def get_even_clusters(X, cluster_size):
    n_clusters = int(np.ceil(len(X) / cluster_size))
    # random_state required for deterministic results.
    kmeans = KMeans(n_clusters, random_state=42)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    centers = (
        centers.reshape(-1, 1, X.shape[-1])
        .repeat(cluster_size, 1)
        .reshape(-1, X.shape[-1])
    )
    distance_matrix = cdist(X, centers)
    clusters = linear_sum_assignment(distance_matrix)[1] // cluster_size
    return list(clusters)


def kmeans_model(embedding_model):
    def inner(words: List[str]) -> List[List[str]]:
        embeddings = embedding_model.encode(words)
        embeddings = embeddings - embeddings.mean(axis=1, keepdims=True)
        embeddings = embeddings / embeddings.std(axis=1, keepdims=True)
        res = get_even_clusters(embeddings, 4)
        result = {}
        for word_i, cluster in enumerate(res):
            if cluster not in result:
                result[cluster] = []
            result[cluster].append(words[word_i])
        return list(result.values())

    return inner


class KMeansModel:
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2")
        # self.embedding_model.cpu()  # Required when using multiprocessing

    def run(self, words: List[str]):
        embeddings = self.embedding_model.encode(words)
        embeddings = embeddings - embeddings.mean(axis=1, keepdims=True)
        embeddings = embeddings / embeddings.std(axis=1, keepdims=True)
        res = get_even_clusters(embeddings, 4)
        result = {}
        for word_i, cluster in enumerate(res):
            if cluster not in result:
                result[cluster] = []
            result[cluster].append(words[word_i])
        return list(result.values())
