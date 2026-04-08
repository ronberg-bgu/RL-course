"""
pddl_to_map.py — Reconstructs an ASCII map from PDDL domain + problem files.

Character codes (matches the simulator):
  W  Wall
  A  Agent
  B  Small box
  C  Heavy box
  G  Goal
     Empty cell
"""

import re


def parse_pddl_to_map(domain_path, problem_path):
    """
    Read problem.pddl and reconstruct the ASCII map used by the simulator.

    The grid dimensions are inferred from the highest loc_X_Y indices found in
    the :objects section.  Any (X, Y) pair that is absent from :objects is
    treated as a wall.

    Returns a list of equal-length strings, one per row (top to bottom).
    """
    with open(problem_path) as f:
        text = f.read()

    # Locate the three major sections we need.
    init_start = text.lower().find('(:init')
    goal_start = text.lower().find('(:goal')

    if init_start == -1:
        raise ValueError("Could not find (:init section in problem.pddl")
    if goal_start == -1:
        raise ValueError("Could not find (:goal section in problem.pddl")

    objects_text = text[:init_start]
    init_text    = text[init_start:goal_start]
    goal_text    = text[goal_start:]

    # ------------------------------------------------------------------
    # 1. Collect all walkable locations from :objects
    # ------------------------------------------------------------------
    loc_coords = set()
    for m in re.finditer(r'loc_(\d+)_(\d+)', objects_text):
        loc_coords.add((int(m.group(1)), int(m.group(2))))

    if not loc_coords:
        raise ValueError("No loc_X_Y locations found in :objects section")

    max_x = max(x for x, y in loc_coords)
    max_y = max(y for x, y in loc_coords)

    # ------------------------------------------------------------------
    # 2. Build grid: (max_y + 2) rows x (max_x + 2) cols, all walls
    # ------------------------------------------------------------------
    grid = [['W'] * (max_x + 2) for _ in range(max_y + 2)]

    # Mark every location that exists in :objects as an empty cell.
    for x, y in loc_coords:
        grid[y][x] = ' '

    # ------------------------------------------------------------------
    # 3. Mark goal cells (G) — derived from (:goal ...) box positions
    # ------------------------------------------------------------------
    for m in re.finditer(
        r'\((?:box-at|heavybox-at)\s+\S+\s+loc_(\d+)_(\d+)\)',
        goal_text
    ):
        x, y = int(m.group(1)), int(m.group(2))
        if (x, y) in loc_coords:
            grid[y][x] = 'G'

    # ------------------------------------------------------------------
    # 4. Mark initial-state objects (may overlap / override goal cells)
    # ------------------------------------------------------------------

    # Small boxes (B)
    for m in re.finditer(r'\(box-at\s+\S+\s+loc_(\d+)_(\d+)\)', init_text):
        x, y = int(m.group(1)), int(m.group(2))
        if (x, y) in loc_coords:
            grid[y][x] = 'B'

    # Heavy box (C)
    for m in re.finditer(r'\(heavybox-at\s+\S+\s+loc_(\d+)_(\d+)\)', init_text):
        x, y = int(m.group(1)), int(m.group(2))
        if (x, y) in loc_coords:
            grid[y][x] = 'C'

    # Agents (A)
    for m in re.finditer(r'\(agent-at\s+\S+\s+loc_(\d+)_(\d+)\)', init_text):
        x, y = int(m.group(1)), int(m.group(2))
        if (x, y) in loc_coords:
            grid[y][x] = 'A'

    return [''.join(row) for row in grid]


if __name__ == "__main__":
    import os
    pddl_dir = os.path.join(os.path.dirname(__file__), "pddl")
    ascii_map = parse_pddl_to_map(
        os.path.join(pddl_dir, "domain.pddl"),
        os.path.join(pddl_dir, "problem.pddl"),
    )
    print(f"Reconstructed {len(ascii_map)} x {len(ascii_map[0])} map:")
    for row in ascii_map:
        print(f"  {repr(row)}")
