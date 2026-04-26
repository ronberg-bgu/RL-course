"""
pddl_to_map.py — Reconstruct an ASCII map from a PDDL problem file.
"""

import re
import sys
import os


def pddl_to_ascii_map(problem_path):
    """Parse a PDDL problem file and return an ASCII map as a list of strings."""
    text = open(problem_path).read()

    # Locations from :objects block
    objects = re.search(r"\(:objects(.*?)\)", text, re.DOTALL).group(1)
    locations = {(int(m[0]), int(m[1])) for m in re.findall(r"loc_(\d+)_(\d+)", objects)}

    # Entity positions from :init block
    init = re.search(r"\(:init(.*?)\)\s*\(:goal", text, re.DOTALL).group(1)

    def find_positions(pattern):
        return {(int(m[-2]), int(m[-1])) for m in re.findall(pattern, init)}

    agents = find_positions(r"\(agent-at\s+(\w+)\s+loc_(\d+)_(\d+)\)")
    boxes = find_positions(r"\(box-at\s+(\w+)\s+loc_(\d+)_(\d+)\)")
    hboxes = find_positions(r"\(heavybox-at\s+(\w+)\s+loc_(\d+)_(\d+)\)")
    goals = find_positions(r"\(goal\s+loc_(\d+)_(\d+)\)")

    # Build grid — any cell not in locations is a wall
    w = max(x for x, _ in locations) + 2
    h = max(y for _, y in locations) + 2
    grid = []
    for y in range(h):
        row = ""
        for x in range(w):
            if (x, y) not in locations:  row += "W"
            elif (x, y) in agents:       row += "A"
            elif (x, y) in boxes:        row += "B"
            elif (x, y) in hboxes:       row += "C"
            elif (x, y) in goals:        row += "G"
            else:                         row += " "
        grid.append(row)
    return grid


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, "pddl", "problem.pddl")
    for row in pddl_to_ascii_map(path):
        print(row)
