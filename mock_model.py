from typing import List


def mock_model(words: List[str]) -> List[List[str]]:
    return [words[i : i + 4] for i in range(0, len(words), 4)]
