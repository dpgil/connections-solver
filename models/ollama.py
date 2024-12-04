from ollama import generate
import json
from typing import List

SYSTEM_PROMPT_QwQ = """Solve today's NYT Connections game. Here are the instructions for how to play this game:
Find groups of four items that share something in common.

Category Examples:
FISH: Bass, Flounder, Salmon, Trout
FIRE ___: Ant, Drill, Island, Opal

Categories will always be more specific than '5-LETTER-WORDS,' 'NAMES,' or 'VERBS.'

Example 1:
Words: ['DART', 'HEM', 'PLEAT', 'SEAM', 'CAN', 'CURE', 'DRY', 'FREEZE', 'BITE', 'EDGE', 'PUNCH', 'SPICE', 'CONDO', 'HAW', 'HERO', 'LOO']
Groupings:
Things to sew: ['DART', 'HEM', 'PLEAT', 'SEAM']
Ways to preserve food: ['CAN', 'CURE', 'DRY', 'FREEZE']
Sharp quality: ['BITE', 'EDGE', 'PUNCH', 'SPICE']
Birds minus last letter: ['CONDO', 'HAW', 'HERO', 'LOO']

Example 2:
Words: ['COLLECTIVE', 'COMMON', 'JOINT', 'MUTUAL', 'CLEAR', 'DRAIN', 'EMPTY', 'FLUSH', 'CIGARETTE', 'PENCIL', 'TICKET', 'TOE', 'AMERICAN', 'FEVER', 'LUCID', 'PIPE']
Groupings:
Shared: ['COLLECTIVE', 'COMMON', 'JOINT', 'MUTUAL']
Rid of contents: ['CLEAR', 'DRAIN', 'EMPTY', 'FLUSH']
Associated with “stub”: ['CIGARETTE', 'PENCIL', 'TICKET', 'TOE']
__ Dream: ['AMERICAN', 'FEVER', 'LUCID', 'PIPE']

Example 3:
Words: ['HANGAR', 'RUNWAY', 'TARMAC', 'TERMINAL', 'ACTION', 'CLAIM', 'COMPLAINT', 'LAWSUIT', 'BEANBAG', 'CLUB', 'RING', 'TORCH', 'FOXGLOVE', 'GUMSHOE', 'TURNCOAT', 'WINDSOCK']
Groupings:
Parts of an airport: ['HANGAR', 'RUNWAY', 'TARMAC', 'TERMINAL']
Legal terms: ['ACTION', 'CLAIM', 'COMPLAINT', 'LAWSUIT']
Things a juggler juggles: ['BEANBAG', 'CLUB', 'RING', 'TORCH']
Words ending in clothing: ['FOXGLOVE', 'GUMSHOE', 'TURNCOAT', 'WINDSOCK']
Categories share commonalities:

There are 4 categories of 4 words each.
Every word will be in only 1 category.
One word will never be in two categories.
There may be red herrings (words that seem to belong together but actually are in separate categories).
There may be compound words with a common prefix or suffix word.
A few other common categories include word and letter patterns, pop culture clues (such as music and movie titles), and fill-in-the-blank phrases.
You will be given a new example (Example 4) with today's list of words. First, explain your reason for each category and then give your final answer following the structure below (Replace Category 1, 2, 3, 4 with their names instead):

Groupings:
Category1: [word1, word2, word3, word4]
Category2: [word5, word6, word7, word8]
Category3: [word9, word10, word11, word12]
Category4: [word13, word14, word15, word16]

Remember that the same word cannot be repeated across multiple categories, and you need to output 4 categories with 4 distinct words each. Also, do not make up words not in the list. This is the most important rule. Please obey.
All words and explanations must be in English
"""

USER_PROMPT_QwQ = """Example 4:
Words: {words}
Groupings"""

SYSTEM_PROMPT2 ="""
You are a helpful assistant who identifies 4 categories each containing 4 words from the input text.
There are 4 categories of 4 words each.
Every word will be in only 1 category.
One word will never be in two categories.

You will be given an input text with reasoning. First, explain your reason for each category and then give your final answer as JSON following the structure below (Replace Category 1, 2, 3, 4 with their names instead):

Groupings:
Category1: [word1, word2, word3, word4]
Category2: [word5, word6, word7, word8]
Category3: [word9, word10, word11, word12]
Category4: [word13, word14, word15, word16]

Remember that the same word cannot be repeated across multiple categories, and you need to output 4 categories with 4 distinct words each. Also, do not make up words not in the list. This is the most important rule. Please obey.
"""

USER_PROMPT2 = """
{res}

Groupings:
"""

def ollama_model(words: List[str]) -> List[List[str]]:
    # print(f"Trying {words}")
    qwq_response = generate(
        model="qwq",
        prompt=USER_PROMPT_QwQ.format(words=words),
        system=SYSTEM_PROMPT_QwQ,
        options={"temperature": 0, "num_predict": 1024, "min_p": 0.1, "repeat_last_n": 1024, "seed": 0},
        context=[],
        stream=False,
    )

    res = qwq_response["response"]

    resp = generate(
        model="qwq",
        prompt=USER_PROMPT2.format(res = res),
        system=SYSTEM_PROMPT2,
        options={"temperature": 0, "num_predict": 1024, "min_p": 0.1, "repeat_last_n": 1024, "seed": 0},
        context=[],
        format="json",
        stream=False
    )
    raw_response = json.loads(resp["response"])
    outputs = raw_response.values()
    print("\n")
    print(outputs)
    try:
        output_list = list(list(outputs)[0].values())
        output_list = [[o.upper() for o in olist] for olist in output_list]
    except Exception as e:
        print(e)
        output_list = []

    return output_list
