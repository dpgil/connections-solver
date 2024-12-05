import json
import wikipedia as wp
from typing import List
import re
from sentence_transformers import SentenceTransformer
from models.cosine_similarity import cosine_similarity
from models.wordnet import wordnet_group


# Any match between the category element sets.
def categories_match(a, b):
    return bool(a & b)


def wikipedia_group(words, categories) -> List[str]:
    for root in words:
        group = set([root])
        while True:
            best_candidate = None
            for candidate in words:
                if candidate not in group:
                    matches_group = True
                    for word in group:
                        if categories[word].isdisjoint(categories[candidate]):
                            matches_group = False
                    if matches_group:
                        best_candidate = candidate
            if best_candidate is not None:
                group.add(best_candidate)
            else:
                break
        if len(group) == 4:
            return list(group)
    return []


def wikipedia_search(word, search_results):
    results = []
    if word in search_results:
        results = search_results[word]
    else:
        results = wp.search(word, results=20)
    return results


def wikipedia_categories(words: List[str], search_results):
    categories = {}
    for word in words:
        results = wikipedia_search(word, search_results)
        # Grab the "categories" from the title of relevant wikipedia pages.
        # ["Chicago", "Chicago (musical)", "Chicago Bears", "Chicago (hot dog)"] => set(["musical", "hot dog"])
        parenthetical_contents = {
            match.group(1) for s in results if (match := re.search(r"\(([^)]+)\)", s))
        }
        # Common wikipedia category that doesn't mean anything.
        parenthetical_contents.discard("disambiguation")
        categories[word] = parenthetical_contents
    return categories


# Cached wikipedia search results for faster iteration.
def load_search_results():
    with open("data/wikipedia.json", "r") as file:
        data = json.load(file)

    # Create a dictionary that maps each word to its results
    results = {entry["word"]: entry["results"] for entry in data}
    return results


class WikipediaModel:
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2")
        self.search_results = load_search_results()

    def get_group(self, words: List[str], categories) -> List[str]:
        group = wikipedia_group(words, categories)
        if not group:
            group = wordnet_group(words)
        if not group:
            embeddings = self.embedding_model.encode(words)
            group = cosine_similarity(embeddings, words)
        return group

    def run(self, words: List[str]) -> List[List[str]]:
        groups = []
        categories = wikipedia_categories(words, self.search_results)
        while words:
            group = self.get_group(words, categories)
            groups.append(group)
            for word in group:
                words.remove(word)
        return groups
