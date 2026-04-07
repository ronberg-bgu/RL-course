import sys
import os
import re

# Ensure the root directory is in the path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from llm_pipeline import generate_pddl_with_gemini, extract_and_save_pddl, PROMPT
from pddl_to_map import parse_pddl_to_map
from visualize_plan import visualize_pddl_plan


# --- TEE CLASS FOR LOGGING ---
class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def run_live_demo():
    # Create the output file
    log_file = open("exercises/ex1/planner_output.txt", "w", encoding="utf-8")

    # Redirect standard output and error to both the console and the file
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    try:
        print("\n" + "=" * 50)
        print("🤖 RL COURSE EX1: LIVE DEMO PIPELINE")
        print("=" * 50)

        # PHASE 1: LLM Generation
        print("\n▶️ PHASE 1: LLM Generation")
        llm_output = generate_pddl_with_gemini(PROMPT)
        extract_and_save_pddl(llm_output)

        # PHASE 2: PDDL to ASCII Map Translation
        print("\n▶️ PHASE 2: PDDL to ASCII Map Translation")
        ascii_map = parse_pddl_to_map("exercises/ex1/pddl/problem.pddl")

        if not ascii_map:
            print("❌ Map parsing failed. Exiting.")
            return

        print("\n🗺️ Generated Map Layout:")
        for row in ascii_map:
            print(row)

        # PHASE 3: Solving & Visualization
        print("\n▶️ PHASE 3: Solving & Visualization")
        visualize_pddl_plan(ascii_map, "exercises/ex1/pddl/domain.pddl", "exercises/ex1/pddl/problem.pddl")

    finally:
        # Restore original streams and close file
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
        print("\n✅ Full log saved to planner_output.txt")


if __name__ == "__main__":
    run_live_demo()

# python .\exercises\ex1\main.py
