from simulator import check_attempt
import unittest


class TestVerifyResult(unittest.TestCase):
    def test_check_attempt(self):
        answer = [
            ["a", "b", "c", "d"],
            ["e", "f", "g", "h"],
            ["i", "j", "k", "l"],
            ["m", "n", "o", "p"],
        ]
        attempt = [
            ["a", "b", "c", "d"],
            ["e", "f", "g", "h"],
            ["i", "j", "k", "m"],  # m and l swapped
            ["l", "n", "o", "p"],
        ]
        matching_count = check_attempt(answer, attempt)
        self.assertEqual(matching_count, 2)

    def test_check_attempt_reordered(self):
        answer = [
            ["a", "b", "c", "d"],
            ["e", "f", "g", "h"],
            ["i", "j", "k", "l"],
            ["m", "n", "o", "p"],
        ]
        attempt = [
            ["i", "j", "k", "m"],  # m and l swapped
            ["l", "n", "o", "p"],
            ["d", "c", "b", "a"],  # reverse order
            ["e", "f", "g", "h"],
        ]
        matching_count = check_attempt(answer, attempt)
        self.assertEqual(matching_count, 2)

    def test_sanity(self):
        self.assertEqual(1, 1)


if __name__ == "__main__":
    unittest.main()
