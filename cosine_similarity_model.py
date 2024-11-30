from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from typing import List
import numpy as np
import itertools


class CosineSimilarityModel:
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2")

    # run_once takes the 4 words it's most confident about
    def run_once(self, embeddings, words: List[str]):
        n = len(words)
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    similarity_matrix[i, j] = 1 - cosine(embeddings[i], embeddings[j])

        # Find the 4 words with the highest mutual similarity
        best_combination = None
        best_score = -float("inf")
        for indices in itertools.combinations(range(n), 4):
            score = sum(
                similarity_matrix[i][j] for i, j in itertools.combinations(indices, 2)
            )
            if score > best_score:
                best_score = score
                best_combination = indices
        return [words[i] for i in best_combination]

    def run(self, words: List[str]):
        clusters = []

        while words:
            embeddings = self.embedding_model.encode(words)
            cluster = self.run_once(embeddings, words)
            clusters.append(cluster)
            for word in cluster:
                words.remove(word)

        return clusters
