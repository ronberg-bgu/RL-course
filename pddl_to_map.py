from pathlib import Path
import re


LOCATION_RE = re.compile(r"loc_(\d+)_(\d+)")
AGENT_RE = re.compile(r"\(agent-at\s+(agent_\d+)\s+(loc_\d+_\d+)\)")
BOX_RE = re.compile(r"\(box-at\s+(box_\d+)\s+(loc_\d+_\d+)\)")
HEAVY_RE = re.compile(r"\(heavybox-at\s+(hbx_\d+)\s+(loc_\d+_\d+)\)")
GOAL_RE = re.compile(r"\((?:box-at|heavybox-at)\s+[\w_]+\s+(loc_\d+_\d+)\)")


def _loc_to_xy(loc_name: str) -> tuple[int, int]:
    match = LOCATION_RE.fullmatch(loc_name)
    if not match:
        raise ValueError(f"Invalid location name: {loc_name}")
    return int(match.group(1)), int(match.group(2))


def _extract_goal_section(problem_text: str) -> str:
    start = problem_text.find("(:goal")
    if start == -1:
        return ""

    depth = 0
    for idx in range(start, len(problem_text)):
        char = problem_text[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return problem_text[start : idx + 1]

    raise ValueError("Could not parse the goal section in problem.pddl")


def _extract_init_section(problem_text: str) -> str:
    start = problem_text.find("(:init")
    if start == -1:
        return ""

    depth = 0
    for idx in range(start, len(problem_text)):
        char = problem_text[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return problem_text[start : idx + 1]

    raise ValueError("Could not parse the init section in problem.pddl")


def parse_pddl_to_map(domain_path: str | Path, problem_path: str | Path) -> list[str]:
    _ = Path(domain_path).read_text(encoding="utf-8")
    problem_text = Path(problem_path).read_text(encoding="utf-8")
    init_text = _extract_init_section(problem_text)
    goal_text = _extract_goal_section(problem_text)

    locations = {(int(x), int(y)) for x, y in LOCATION_RE.findall(problem_text)}
    if not locations:
        raise ValueError("No locations were found in problem.pddl")

    max_x = max(x for x, _ in locations)
    max_y = max(y for _, y in locations)
    grid = [["W" for _ in range(max_x + 1)] for _ in range(max_y + 1)]

    for x, y in locations:
        grid[y][x] = " "

    goals: set[tuple[int, int]] = set()
    for loc_name in GOAL_RE.findall(goal_text):
        goals.add(_loc_to_xy(loc_name))

    occupants: dict[tuple[int, int], str] = {}

    for _, loc_name in BOX_RE.findall(init_text):
        x, y = _loc_to_xy(loc_name)
        occupants[(x, y)] = "B"

    for _, loc_name in HEAVY_RE.findall(init_text):
        x, y = _loc_to_xy(loc_name)
        occupants[(x, y)] = "C"

    for _, loc_name in AGENT_RE.findall(init_text):
        x, y = _loc_to_xy(loc_name)
        occupants[(x, y)] = "A"

    for x, y in locations:
        if (x, y) in goals:
            # Keep goals visible in the reconstructed map even if occupied in init.
            grid[y][x] = "G"
        elif (x, y) in occupants:
            grid[y][x] = occupants[(x, y)]

    return ["".join(row) for row in grid]
