from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util
from itertools import combinations
import numpy as np
from typing import List
from cosine_similarity_model import cosine_similarity


def word_net_similarity(word1: str, word2: str) -> int:
    max_sim = 0
    for synset1 in wordnet.synsets(word1):
        for synset2 in wordnet.synsets(word2):
            max_sim = max(max_sim, synset1.wup_similarity(synset2))
    return max_sim


# Greedily returns a group of 4 words it's confident about, or empty list if it doesn't find one.
def word_net(words: List[str]) -> List[str]:
    used = set()
    for word in words:
        if word in used:
            continue
        group = [word]
        for other_word in words:
            if word != other_word:
                max_sim = word_net_similarity(word, other_word)
                if max_sim >= 0.88:  # Guessing a threshold here
                    group.append(other_word)
        if len(group) == 4:
            return group
    return []


class WordNetModel:
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2")

    def run(self, words: List[str]) -> List[List[str]]:
        groups = []
        while words:
            group = word_net(words)
            if not group:
                # Fall back to cosine similarity
                embeddings = self.embedding_model.encode(words)
                group = cosine_similarity(embeddings, words)
            groups.append(group)
            for word in group:
                words.remove(word)
        return groups
