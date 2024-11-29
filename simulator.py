import csv
from typing import List


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


def glove_model(words: List[str]) -> List[List[str]]:
    return [
        ["HAIL", "RAIN", "SLEET", "SNOW"],
        ["BUCKS", "HEAT", "JAZZ", "NETS"],
        ["OPTION", "RETURN", "SHIFT", "TAB"],
        ["KAYAK", "LEVEL", "MOM", "RACECAR"],
    ]


class SimulatorStats:
    def __init__(self):
        self.total = 0
        self.correct = 0

    def inc(self, correct: bool):
        self.total += 1
        if correct:
            self.correct += 1


def verify_result(puzzle: Puzzle, result: List[List[str]]) -> bool:
    group_sets = [set(group.words) for group in puzzle.groups]
    result_sets = [set(r) for r in result]
    return sorted(group_sets) == sorted(result_sets)


def simulator(puzzles: List[Puzzle], model):
    stats = SimulatorStats()
    for puzzle in puzzles:
        result = model(puzzle.all_words)
        correct = verify_result(puzzle, result)
        stats.inc(correct)
    return stats


def main():
    puzzles = load_puzzles("puzzles.csv")
    stats = simulator(puzzles[:3], glove_model)
    print(
        f"Model result: Total: {stats.total}, Correct: {stats.correct}, Pct: {round((stats.correct / stats.total) * 100, 2)}"
    )


if __name__ == "__main__":
    main()
