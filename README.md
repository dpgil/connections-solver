# Connections Solver

Connections Solver is a framework for building models to solve [NYT Connections](https://www.nytimes.com/games/connections) puzzles.
These puzzles have been shown to be a meaningful benchmark for the capabilities of LLMs ([paper](https://arxiv.org/pdf/2406.11012)).

This project includes a simulator which loads all historical NYT Connections puzzles, runs them against a given model,
and outputs the performance of the model. It also includes various models in attempt to beat existing benchmarks.

## How to use

Run the simulator with `python3 simulator.py`.

The simulator will run the number of puzzles specified with the flag `-n <num_puzzles>`
and output stats based on the number of puzzles and individual groups the given model got right.

Switch the model type with the flag `--model <model_name>`, where the model name is one of
`cosine_similarity`, `kmeans`, `wordnet`, `ollama`, or `mock`.

Run the simulator with `--debug` for comparisons between the puzzle answer and the model output.

Run the simulator with `--parallel` to parallelize puzzle solving. Note: some ML libraries don't work well with parallelism
so this may actually slow down execution.

Use `--offset <num>` to start from a different puzzle in the list.

To implement a new model, implement a function that takes a list of 16 words
and outputs a `List[List[str]]`, which is the list of groups for the attempted solution
where each group contains 4 words.

**Note: if you see something like ModuleNotFound error, run `source venv/bin/activate`**

## Dataset

The dataset for NYT Connections puzzles is pulled from
https://huggingface.co/datasets/eric27n/NYT-Connections.

## Unit tests

To run unit tests, run `python3 -m unittest discover -p "*_test.py"`.
