import re
from pathlib import Path

def generate_ascii_map(pddl_file_path):
    try:
        content = Path(pddl_file_path).read_text()
    except FileNotFoundError:
        print(f"Error: {pddl_file_path} not found.")
        return

    locations = re.findall(r'loc_(\d+)_(\d+)', content)
    if not locations:
        print("No locations found in PDDL.")
        return

    coords = [(int(x), int(y)) for x, y in locations]
    max_x = max(c[0] for c in coords)
    max_y = max(c[1] for c in coords)

    grid = [[" " for _ in range(max_x + 2)] for _ in range(max_y + 2)]

    for y in range(max_y + 2):
        for x in range(max_x + 2):
            if x == 0 or y == 0 or x == max_x + 1 or y == max_y + 1:
                grid[y][x] = "W"

    clears = set(re.findall(r'\(clear loc_(\d+)_(\d+)\)', content))
    agents_init = set(re.findall(r'\(agent-at agent_\d+ loc_(\d+)_(\d+)\)', content))
    boxes_init = set(re.findall(r'\(box-at box_\d+ loc_(\d+)_(\d+)\)', content))
    heavy_init = set(re.findall(r'\(heavybox-at hbx_\d+ loc_(\d+)_(\d+)\)', content))
    
    occupied_or_clear = clears | agents_init | boxes_init | heavy_init
    for x, y in coords:
        if (str(x), str(y)) not in occupied_or_clear:
            grid[int(y)][int(x)] = "W"

    for x, y in agents_init:
        grid[int(y)][int(x)] = "A"

    for x, y in boxes_init:
        grid[int(y)][int(x)] = "B"

    for x, y in heavy_init:
        grid[int(y)][int(x)] = "C"

    goal_section = re.search(r'\(:goal(.*?)\)\s*\)', content, re.DOTALL)
    if goal_section:
        goal_content = goal_section.group(1)
        goal_locs = re.findall(r'loc_(\d+)_(\d+)', goal_content)
        for x, y in goal_locs:
            grid[int(y)][int(x)] = "G"

    print("large_map = [")
    for i, row in enumerate(grid):
        row_str = "".join(row)
        comma = "," if i < len(grid) - 1 else ""
        print(f'    "{row_str}"{comma}')
    print("]")

    final_ascii_list = []
    for row in grid:
        final_ascii_list.append("".join(row))
    
    return final_ascii_list

if __name__ == "__main__":
    generate_ascii_map("pddl/problem.pddl")