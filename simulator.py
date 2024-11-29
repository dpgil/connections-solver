import csv
from typing import List
from kmeans_model import kmeans_model


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

    def inc(self, matching_groups: int):
        self.total += 1
        self.matching_groups += matching_groups
        if matching_groups == 4:
            self.correct += 1


def simulator(puzzles: List[Puzzle], model) -> SimulatorStats:
    stats = SimulatorStats()
    for puzzle in puzzles:
        result = model(puzzle.all_words)
        matching_groups = check_attempt(puzzle.get_answers(), result)
        stats.inc(matching_groups)
    return stats


def main():
    puzzles = load_puzzles("puzzles.csv")
    stats = simulator(puzzles[:1], kmeans_model)

    correct_puzzles = stats.correct
    total_puzzles = stats.total
    puzzle_pct = round(correct_puzzles / total_puzzles * 100, 2)

    correct_groups = stats.matching_groups
    total_groups = total_puzzles * 4
    group_pct = round(correct_groups / total_groups * 100, 2)

    print(f"Model result")
    print(
        f"Total puzzles: {total_puzzles}, Correct puzzles: {correct_puzzles}, Pct: {puzzle_pct}"
    )
    print(
        f"Total groups: {total_groups}, Correct groups: {correct_groups}, Pct: {group_pct}"
    )


if __name__ == "__main__":
    main()
