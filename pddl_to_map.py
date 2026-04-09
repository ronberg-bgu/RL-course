import re



def extract_goal_locations(pddl_text):
    """
    Reads a PDDL file and extracts all location names in the :goal section.
    Works regardless of indentation, spaces, or line endings.
    """
    with open(pddl_text, 'r', encoding='utf-8') as f:
        text = f.read()

    # Normalize whitespace
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Find the position of :goal (ignore leading spaces)
    goal_pos = text.find(":goal")
    if goal_pos == -1:
        print("No :goal found in file!")
        return []

    # Take everything after :goal
    goal_text = text[goal_pos:]

    # Find all loc_x_y patterns
    locations = re.findall(r'loc_\d+_\d+', goal_text)

    return locations

def parse_pddl(file_path):
    with open(file_path, "r") as f:
        text = f.read()

    predicates = re.findall(r"\(([^()]+)\)", text)

    agents = {}
    boxes = {}
    heavy_boxes = {}
    goals = set()

    # Extract all locations
    all_locations = set(re.findall(r"loc_\d+_\d+", text))

    # Extract init section
    init_section = re.search(r"\(:init(.*?)\)\s*\(:goal", text, re.DOTALL)
    if init_section:
        init_preds = re.findall(r"\(([^()]+)\)", init_section.group(1))
        for pred in init_preds:
            tokens = pred.split()
            if len(tokens) < 2:
                continue
            if tokens[0] == "agent-at":
                agents[tokens[1]] = tokens[2]
            elif tokens[0] == "box-at":
                boxes[tokens[1]] = tokens[2]
            elif tokens[0] == "heavybox-at":
                heavy_boxes[tokens[1]] = tokens[2]
            

    # Extract goal section
    
    goal_preds = extract_goal_locations(file_path)
    goals = goal_preds
    #print(goal_preds)
    return agents, boxes, heavy_boxes, goals, all_locations


def extract_grid_size(locations):
    max_x = 0
    max_y = 0

    for loc in locations:
        match = re.match(r"loc_(\d+)_(\d+)", loc)
        if match:
            x, y = map(int, match.groups())
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    return max_x+2, max_y+2


def build_ascii_map(agents, boxes, heavy_boxes, goals, all_locations):
    max_x, max_y = extract_grid_size(all_locations)

    # Initialize grid with spaces
    grid = [[" " for _ in range(max_x)] for _ in range(max_y)]

    def get_coords(loc):
        match = re.match(r"loc_(\d+)_(\d+)", loc)
        if not match:
            return None
        x, y = map(int, match.groups())
        return y, x   # flip Y

    # 🔥 AUTO WALLS: border only
    for y in range(max_y):
        for x in range(max_x):
            if y == 0 or y == max_y - 1 or x == 0 or x == max_x - 1:
                grid[y][x] = "W"

    # Place goals
    for loc in goals:
        coords = get_coords(loc)
        if coords:
            y, x = coords
            grid[y][x] = "G"

    # Helper for overlap
    def place_with_goal(loc, normal, goal_char):
        coords = get_coords(loc)
        if not coords:
            return
        y, x = coords
        if grid[y][x] == "G":
            grid[y][x] = goal_char
        elif grid[y][x] != "W":
            grid[y][x] = normal

    # Place objects
    for loc in boxes.values():
        place_with_goal(loc, "B", "b")

    for loc in heavy_boxes.values():
        place_with_goal(loc, "C", "h")

    for loc in agents.values():
        place_with_goal(loc, "A", "a")

    return grid


def print_map(grid):
    for row in grid:
        print("".join(row))

def parse_pddl_to_map(problem_file_path):
    

    agents, boxes, heavy_boxes, goals, all_locations = parse_pddl(problem_file_path)

    grid = build_ascii_map(
        agents, boxes, heavy_boxes, goals, all_locations
    )

    return grid

def main():
    file_path = "pddl/problem_v2.pddl"

    agents, boxes, heavy_boxes, goals, all_locations = parse_pddl(file_path)

    grid = build_ascii_map(
        agents, boxes, heavy_boxes, goals, all_locations
    )

    print_map(grid)


if __name__ == "__main__":
    main()