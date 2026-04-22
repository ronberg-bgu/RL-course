"""
pddl_to_map.py — Parse PDDL domain + problem files back into an ASCII map.

Reads the generated PDDL files and reconstructs the ASCII representation
used by the MultiAgentBoxPushEnv simulator.

ASCII character mapping:
    W  — Wall
    A  — Agent
    B  — Small box
    C  — Heavy box
    G  — Goal cell
    ' ' — Empty cell
"""

import re
import sys
import os


def _extract_section(text, keyword):
    """
    Extract the content of a PDDL section like (:init ...) or (:goal ...).
    Handles nested parentheses by counting open/close parens.
    """
    start = text.find(f"(:{keyword}")
    if start == -1:
        return ""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


def parse_pddl_to_map(domain_path, problem_path):
    """
    Parse a PDDL problem file and reconstruct the ASCII map.

    Parameters
    ----------
    domain_path : str
        Path to the PDDL domain file (used only for validation).
    problem_path : str
        Path to the PDDL problem file containing init predicates.

    Returns
    -------
    list[str]
        ASCII map as a list of strings, one per row.
    """
    with open(problem_path, "r") as f:
        problem_text = f.read()

    # ---- Split into init and goal sections (balanced-paren extraction) ----
    init_text = _extract_section(problem_text, "init")
    goal_text = _extract_section(problem_text, "goal")

    # ---- Extract all location names to determine grid dimensions ----
    loc_pattern = re.compile(r"loc_(\d+)_(\d+)")
    all_locs = set()
    for match in loc_pattern.finditer(problem_text):
        x, y = int(match.group(1)), int(match.group(2))
        all_locs.add((x, y))

    if not all_locs:
        raise ValueError("No locations found in PDDL problem file.")

    max_x = max(x for x, y in all_locs)
    max_y = max(y for x, y in all_locs)

    # Grid dimensions include the outer wall border
    width = max_x + 2
    height = max_y + 2

    # ---- Initialize grid: walls everywhere, then carve out known locations ----
    grid = [["W"] * width for _ in range(height)]

    for x, y in all_locs:
        grid[y][x] = " "

    # ---- Parse agent positions (from :init only) ----
    agent_pattern = re.compile(r"\(agent-at\s+(agent_\d+)\s+loc_(\d+)_(\d+)\)")
    for match in agent_pattern.finditer(init_text):
        x, y = int(match.group(2)), int(match.group(3))
        grid[y][x] = "A"

    # ---- Parse small box positions (from :init only) ----
    box_pattern = re.compile(r"\(box-at\s+box_\d+\s+loc_(\d+)_(\d+)\)")
    for match in box_pattern.finditer(init_text):
        x, y = int(match.group(1)), int(match.group(2))
        grid[y][x] = "B"

    # ---- Parse heavy box positions (from :init only) ----
    heavybox_pattern = re.compile(r"\(heavybox-at\s+hbx_\d+\s+loc_(\d+)_(\d+)\)")
    for match in heavybox_pattern.finditer(init_text):
        x, y = int(match.group(1)), int(match.group(2))
        grid[y][x] = "C"

    # ---- Parse goal positions (from :goal section) ----
    goal_loc_pattern = re.compile(r"(?:box-at|heavybox-at)\s+\S+\s+loc_(\d+)_(\d+)")
    for match in goal_loc_pattern.finditer(goal_text):
        x, y = int(match.group(1)), int(match.group(2))
        # Only mark as goal if nothing else is placed there initially
        if grid[y][x] == " ":
            grid[y][x] = "G"

    ascii_map = ["".join(row) for row in grid]
    return ascii_map


if __name__ == "__main__":
    # Default paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    domain_default = os.path.join(script_dir, "pddl", "domain.pddl")
    problem_default = os.path.join(script_dir, "pddl", "problem.pddl")

    domain_path = sys.argv[1] if len(sys.argv) > 1 else domain_default
    problem_path = sys.argv[2] if len(sys.argv) > 2 else problem_default

    ascii_map = parse_pddl_to_map(domain_path, problem_path)

    print("Reconstructed ASCII map:")
    print("-" * 40)
    for row in ascii_map:
        print(row)
    print("-" * 40)
