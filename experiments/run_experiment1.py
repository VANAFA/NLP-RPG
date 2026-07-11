"""Single entrypoint for Experiment 1: generate -> judge -> aggregate.

Run the three steps standalone (run_generate.py / judge.py / aggregate.py)
instead of this if you want to, e.g., re-judge or re-aggregate without
re-generating (useful after tweaking the judge rubric).
"""

import aggregate
import judge
import run_generate


def main():
    run_generate.main()
    judge.main()
    aggregate.main()


if __name__ == "__main__":
    main()
