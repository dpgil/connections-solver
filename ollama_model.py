import ollama
import json
from typing import List


def ollama_model(words: List[str]) -> List[List[str]]:
    resp = ollama.generate(
        model="llama3.2",
        prompt=(
            f"You are solving a puzzle similar to the New York Times Connections game. "
            f"In this game, you are given 16 words and you must organize them into 4 distinct groups. "
            f"Each group contains 4 words, and the words in each group have something in common. "
            f"For example: "
            f"- The words BASS, FLOUNDER, SALMON, TROUT form a group because they are all types of fish. "
            f"- The words BUCKS, HEAT, JAZZ, NETS form a group because they are all NBA team mascots. "
            f"Your task is to group the following 16 words into 4 groups of 4 words each. "
            f"The groups should be based on any shared characteristics the words might have. "
            f"Please output the groups as a list of lists in the following exact format (comma-separated and with double quotes), and with no other text: [[\"word1\", \"word2\", ...], [\"word5\", \"word6\", ...], ...]. "
            f"Here are the 16 words: {words}"
        ),
    )
    raw_response = resp['response']
    cleaned_response = '[' + raw_response.replace("\n", "") + ']'
    parsed_data = json.loads(cleaned_response)
    return parsed_data
