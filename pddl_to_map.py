from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


Loc = Tuple[int, int]


def extract_section(text: str, section_name: str) -> str:
    """
    Extract a top-level PDDL section like (:init ...) or (:goal ...).
    Uses simple parenthesis balancing from the section start.
    """
    start_token = f"(:{section_name}"
    start = text.find(start_token)
    if start == -1:
        return ""

    depth = 0
    end = start
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    return text[start:end]


def parse_locations_from_objects(text: str) -> Set[Loc]:
    """
    Parse all declared locations from the whole file.
    """
    return {(int(x), int(y)) for x, y in re.findall(r"\bloc_(\d+)_(\d+)\b", text)}


def parse_init_positions(init_text: str) -> tuple[Dict[str, Loc], Dict[str, Loc], Dict[str, Loc]]:
    """
    Parse only initial positions from the :init section.
    """
    agents: Dict[str, Loc] = {}
    boxes: Dict[str, Loc] = {}
    heavyboxes: Dict[str, Loc] = {}

    for name, x, y in re.findall(r"\(agent-at\s+([A-Za-z0-9_]+)\s+loc_(\d+)_(\d+)\)", init_text):
        agents[name] = (int(x), int(y))

    for name, x, y in re.findall(r"\(box-at\s+([A-Za-z0-9_]+)\s+loc_(\d+)_(\d+)\)", init_text):
        boxes[name] = (int(x), int(y))

    for name, x, y in re.findall(r"\(heavybox-at\s+([A-Za-z0-9_]+)\s+loc_(\d+)_(\d+)\)", init_text):
        heavyboxes[name] = (int(x), int(y))

    return agents, boxes, heavyboxes


def parse_goal_locations(goal_text: str) -> Set[Loc]:
    """
    Parse only goal cells from the :goal section.
    """
    goals: Set[Loc] = set()

    for _, x, y in re.findall(r"\(box-at\s+([A-Za-z0-9_]+)\s+loc_(\d+)_(\d+)\)", goal_text):
        goals.add((int(x), int(y)))

    for _, x, y in re.findall(r"\(heavybox-at\s+([A-Za-z0-9_]+)\s+loc_(\d+)_(\d+)\)", goal_text):
        goals.add((int(x), int(y)))

    return goals


def build_ascii_map(
    all_locations: Set[Loc],
    agents: Dict[str, Loc],
    boxes: Dict[str, Loc],
    heavyboxes: Dict[str, Loc],
    goal_locations: Set[Loc],
) -> List[str]:
    """
    Build ASCII map with outer wall border.
    Priority:
    G first, then B/C/A override it visually.
    """
    if not all_locations:
        raise ValueError("No locations found in problem file.")

    max_x = max(x for x, _ in all_locations)
    max_y = max(y for _, y in all_locations)

    rows: List[str] = []

    for y in range(0, max_y + 2):
        row_chars: List[str] = []
        for x in range(0, max_x + 2):
            if x == 0 or y == 0 or x == max_x + 1 or y == max_y + 1:
                row_chars.append("W")
                continue

            loc = (x, y)

            if loc not in all_locations:
                row_chars.append("W")
                continue

            ch = " "
            if loc in goal_locations:
                ch = "G"
            if loc in boxes.values():
                ch = "B"
            if loc in heavyboxes.values():
                ch = "C"
            if loc in agents.values():
                ch = "A"

            row_chars.append(ch)

        rows.append("".join(row_chars))

    return rows


def save_ascii_map(lines: List[str], output_path: Path) -> None:
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    problem_path = repo_root / "pddl" / "problem.pddl"
    output_path = repo_root / "ascii_map.txt"

    if not problem_path.exists():
        print(f"Could not find: {problem_path}")
        return 1

    text = problem_path.read_text(encoding="utf-8")
    text = re.sub(r";.*", "", text)

    all_locations = parse_locations_from_objects(text)
    init_text = extract_section(text, "init")
    goal_text = extract_section(text, "goal")

    agents, boxes, heavyboxes = parse_init_positions(init_text)
    goal_locations = parse_goal_locations(goal_text)

    ascii_lines = build_ascii_map(all_locations, agents, boxes, heavyboxes, goal_locations)

    print("=== ASCII MAP ===")
    for line in ascii_lines:
        print(line)
    print("=================")

    save_ascii_map(ascii_lines, output_path)
    print(f"Saved ASCII map to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())