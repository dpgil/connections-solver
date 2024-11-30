import argparse
import csv
import concurrent.futures
from typing import List
import time
from models.kmeans import KMeansModel
from models.wordnet import WordNetModel
from models.mock import mock_model
from models.cosine_similarity import CosineSimilarityModel
from models.ollama import ollama_model


class Group:
    def __init__(self):
        self.words = []
        self.name = ""
        self.level = -1

    def add_word(self, name: str, level: int, word: str):
        if self.name != "" and name != self.name:
            raise Exception("adding word from a different group to this group")
        if self.level != -1 and level != self.level:
            raise Exception("adding word from a different level to this group")

        self.name = name
        self.level = level
        self.words.append(word)
        self.words.sort()

    def __str__(self):
        return f"Group name: {self.name}, Words: {', '.join(self.words)}"


class Puzzle:
    def __init__(self):
        self.date = ""
        self.all_words = []
        self.groups = [Group(), Group(), Group(), Group()]

    def add_word(self, date: str, group_name: str, level: int, word: str):
        if self.date != "" and date != self.date:
            raise Exception("adding word from a different date to this puzzle")
        self.date = date
        self.all_words.append(word)
        self.groups[level].add_word(group_name, level, word)

    def get_answers(self):
        return [group.words for group in self.groups]

    def __str__(self):
        group_str = ", ".join(str(group) for group in self.groups)
        return f"Date: {self.date}, Groups: {group_str}"


def load_puzzles(filename: str) -> List[Puzzle]:
    with open(filename, newline="") as csvfile:
        reader = csv.DictReader(csvfile)

        puzzles = {}
        for row in reader:
            date = row["Puzzle Date"]
            if date not in puzzles:
                puzzles[date] = Puzzle()
            word = row["Word"]
            group_name = row["Group Name"]
            level = int(row["Group Level"])
            puzzles[date].add_word(date, group_name, level, word)
        return list(puzzles.values())
    raise Exception("could not load puzzles")


# Returns the number of attempt groups that match the answer groups.
# If the value is 4, the attempt is correct.
def check_attempt(answer: List[List[str]], attempt: List[List[str]]) -> int:
    if len(attempt) != 4:
        raise Exception("invalid attempt, expected 4 groups")
    if any(len(group) != 4 for group in attempt):
        raise Exception("invalid group length, expected 4 words in each group")
    answer_sets = [set(group) for group in answer]
    attempt_sets = [set(group) for group in attempt]
    matching_count = sum(1 for s1 in answer_sets if s1 in attempt_sets)
    return matching_count


class SimulatorStats:
    def __init__(self):
        self.total = 0
        self.correct = 0
        self.matching_groups = 0
        self.invalid_attempts = 0

    def invalid_attempt(self):
        self.total += 1
        self.invalid_attempts += 1

    def inc(self, matching_groups: int):
        self.total += 1
        self.matching_groups += matching_groups
        if matching_groups == 4:
            self.correct += 1


def simulator(puzzles: List[Puzzle], model, debug=False) -> SimulatorStats:
    stats = SimulatorStats()
    for puzzle in puzzles:
        result = model(puzzle.all_words)
        try:
            matching_groups = check_attempt(puzzle.get_answers(), result)
        except Exception:
            stats.invalid_attempt()
        else:
            if matching_groups != 4 and debug:
                for inner in result:
                    inner.sort()
                print(f"Answer: {puzzle.get_answers()}\nModel:  {result}")
            stats.inc(matching_groups)
    return stats


def parallel_simulator(puzzles: List[Puzzle], model, debug=False) -> SimulatorStats:
    stats = SimulatorStats()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        inputs = [p.all_words for p in puzzles]
        results = list(executor.map(model, inputs))
        for i, result in enumerate(results):
            # Could parallelize this but I don't think it's computationally expensive.
            matching_groups = check_attempt(puzzles[i].get_answers(), result)
            stats.inc(matching_groups)
        return stats
    raise Exception("error running parallel simulator")


def output_results(secs: int, stats: SimulatorStats):
    print(f"Done, took {secs:.4f} seconds")
    print("------")

    correct_puzzles = stats.correct
    total_puzzles = stats.total
    puzzle_pct = round(correct_puzzles / total_puzzles * 100, 2)

    correct_groups = stats.matching_groups
    total_groups = total_puzzles * 4
    group_pct = round(correct_groups / total_groups * 100, 2)

    print(
        f"Total puzzles: {total_puzzles}, Correct puzzles: {correct_puzzles}, Pct: {puzzle_pct}"
    )
    print(
        f"Total groups: {total_groups}, Correct groups: {correct_groups}, Pct: {group_pct}"
    )
    print(f"Invalid attempts: {stats.invalid_attempts}")


def get_model(model_name: str):
    if model_name == "cosine_similarity":
        return CosineSimilarityModel().run
    elif model_name == "ollama":
        return ollama_model
    elif model_name == "kmeans":
        return KMeansModel().run
    elif model_name == "wordnet":
        return WordNetModel().run
    elif model_name == "mock":
        return mock_model
    else:
        raise Exception("unknown model")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action="store_true", help="Whether to output debug information"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run the simulator in parallel",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=5,
        help="Number of puzzles to run against the model (default 5)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cosine_similarity",
        help='Specify which model to use (default "cosine_similarity")',
    )
    args = parser.parse_args()

    puzzles = load_puzzles("puzzles.csv")
    input = puzzles[: args.n]
    model = get_model(args.model)
    sim_fn = parallel_simulator if args.parallel else simulator
    print(f"Running simulator for {args.n} puzzles with model {args.model}")

    start_time = time.time()
    stats = sim_fn(input, model, debug=args.debug)
    end_time = time.time()

    output_results(end_time - start_time, stats)


if __name__ == "__main__":
    main()
