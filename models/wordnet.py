import itertools
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from models.cosine_similarity import cosine_similarity


def wordnet_similarity(word1: str, word2: str) -> int:
    max_sim = 0
    for synset1 in wordnet.synsets(word1):
        for synset2 in wordnet.synsets(word2):
            max_sim = max(max_sim, synset1.wup_similarity(synset2))
    return max_sim


def compute_similarities(words: List[str]):
    n = len(words)
    similarities = {}
    for pair in itertools.combinations(range(n), 2):
        i, j = pair
        word0 = words[i]
        word1 = words[j]
        similarity = wordnet_similarity(words[i], words[j])
        if word0 not in similarities:
            similarities[word0] = {}
        if word1 not in similarities:
            similarities[word1] = {}
        similarities[word0][word1] = similarity
        similarities[word1][word0] = similarity
    return similarities


# Greedily returns a group of 4 words it's confident about, or empty list if it doesn't find one.
# Builds a group by starting with a root word and adding words that have a similarity above
# some threshold on average with every other word in the group, until no more words are added.
# If the group size is 4, we feel confident about that group and return it.
def wordnet_group(words: List[str]) -> List[str]:
    similarities = compute_similarities(words)

    for root in words:
        group = set([root])
        while True:
            max_sim = 0
            max_candidate = None
            for candidate in words:
                if candidate not in group:
                    curr_sim = 0
                    for word in group:
                        curr_sim += similarities[candidate][word]
                    if curr_sim > max_sim:
                        max_sim = curr_sim
                        max_candidate = candidate
            if max_sim > (0.8 * len(group)):
                group.add(max_candidate)
            else:
                break
        if len(group) == 4:
            return list(group)
    return []


class WordNetModel:
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2")

    def run(self, words: List[str]) -> List[List[str]]:
        groups = []
        while words:
            group = wordnet_group(words)
            if not group:
                # Fall back to cosine similarity to guarantee a group of size 4.
                embeddings = self.embedding_model.encode(words)
                group = cosine_similarity(embeddings, words)
            groups.append(group)
            for word in group:
                words.remove(word)
        return groups
